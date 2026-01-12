#include <iostream>
#include "attn.h"


#include <pybind11/pybind11.h>

// 简单的 C++ 函数
int add(int a, int b) {
    return a + b;
}






// 简单的 C++ 类
class Calculator {
public:
    Calculator(int init_value) : value(init_value) {}
    void add(int x) { value += x; }
    int get() const { return value; }

private:
    int value = 0;
};

// 绑定模块
PYBIND11_MODULE(m_vllm_csrc, m) {
    m.doc() = "pybind11 example plugin"; // 模块文档字符串

    // 绑定函数
    m.def("add", &add, "A function that adds two numbers");

    // 绑定类
    pybind11::class_<Calculator>(m, "Calculator")
        .def(pybind11::init<int>())           // 构造函数
        .def("add", &Calculator::add)         // 成员函数
        .def("get", &Calculator::get);        // 获取值
}

// namespace m_vllm {
//     namespace csrc {
        


//         class Test {
//             public:
//                 Test() {
//                     std::cout << "Test constructor" << std::endl;
//                 }
//         };
//     }
// }


