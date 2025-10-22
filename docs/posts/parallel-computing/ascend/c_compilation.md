由于我常常不懂得怎么去编译C语言，所以这一次借着课程作业的机会，我来好好尝试一下C语言的编译。

```c
cmake_minimum_required(VERSION 3.10)
project(LAB1)
set(CMAKE_C_COMPILER "gcc")

set(SRCS lab1.c)

# 添加 OpenBLAS 的头文件和库路径
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -march=native -fopenmp")
add_executable(test ${SRCS})

# 链接 OpenBLAS
target_link_libraries(test openblas m pthread)
```
这是CMakeLists.txt的内容。

前三行
```c
cmake_minimum_required(VERSION 3.10)  # 要求CMake最低版本
project(LAB1)                          # 项目名称
set(CMAKE_C_COMPILER "gcc")           # 指定使用gcc编译器
```
一个是要求CMake最低版本，一个是指定项目名称以及编译器。这个属于起手式。

第四行开始，最关键的一行。
```c
set(SRCS lab1.c) #源文件列表
```
这告诉你是编译的谁。

