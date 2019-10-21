# build example
add_executable(load1 load1.cpp)
target_link_libraries(load1 nnprobe)
add_executable(probe1 probe1.cpp)
target_link_libraries(probe1 nnprobe)
add_executable(probe2 probe2.cpp)
target_link_libraries(probe2 nnprobe)
add_executable(probe3 probe3.cpp)
target_link_libraries(probe3 nnprobe)
target_link_libraries(probe3 pthread)

# tests
if(TRT)
  set(DEV "-g")
else()
  set(DEV "-c")
endif()

add_test(Loading1 load1 ${DEV})
set_tests_properties (Loading1 PROPERTIES 
   PASS_REGULAR_EXPRESSION "Neural network loaded")

add_test(Probing probe1 ${DEV})
set_tests_properties (Probing PROPERTIES 
   PASS_REGULAR_EXPRESSION "Winning probability")

add_test(Probing_two_networks probe2 ${DEV})
set_tests_properties (Probing_two_networks PROPERTIES 
   PASS_REGULAR_EXPRESSION "Winning probability")

add_test(Probing_with_multiple_threads probe3 ${DEV})
set_tests_properties (Probing_with_multiple_threads PROPERTIES 
   PASS_REGULAR_EXPRESSION "Finished multi-threaded test")

