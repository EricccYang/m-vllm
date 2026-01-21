import xxhash
from typing import TYPE_CHECKING
import numpy as np
from collections import deque

if TYPE_CHECKING:
    from m_vllm.data_classes.batch import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = None
        self.reference_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.reference_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.hash_to_block_id_map: dict[int, int] = dict[int, int]()
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_blocks: deque[int] = deque[int]()
        self.used_block_ids: set[int] = set[int]()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_blocks.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_blocks.append(block_id)

    def can_allocate(self, count: int) -> bool:
        return len(self.free_blocks) >= count

    def may_append(self, seq: "Sequence"):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 0:  # compute block hash
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[seq.block_table[-2]].hash
            hash = self.compute_hash(token_ids, prefix)
            last_block.update(hash, token_ids)
            self.hash_to_block_id_map[hash] = last_block.block_id
        elif len(seq) % self.block_size == 1:  # new
            assert last_block.hash != -1
            new_block_id = self.free_blocks[0]
            self._allocate_block(new_block_id)
            block_table.append(new_block_id)
        else:
            assert last_block.hash == -1

    def allocate(self, seq: "Sequence"):
        cache_miss = False
        h = -1
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            hash_value = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self.hash_to_block_id_map.get(hash_value, -1)
            if block_id == -1 or self.blocks[block_id].hash != hash_value:
                cache_miss = True
            if cache_miss:
                new_block_id = self.free_blocks[0]
                block = self._allocate_block(new_block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.reference_count += 1
                else:
                    block = self._allocate_block(block_id)
            if hash_value != -1:
                block.update(hash_value, token_ids)
                self.hash_to_block_id_map[hash_value] = block.block_id
            seq.block_table.append(block.block_id)

    def deallocate(self, seq: "Sequence"):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
