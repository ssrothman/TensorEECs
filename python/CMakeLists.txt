find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(tensoreecs_py MODULE bindings.cpp)

target_link_libraries(tensoreecs_py PRIVATE tensoreecs)
target_include_directories(tensoreecs_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
set_target_properties(tensoreecs_py PROPERTIES OUTPUT_NAME _tensoreecs)
