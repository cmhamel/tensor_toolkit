if(TENSOR_TOOLKIT_ENABLE_KOKKOS OR TENSOR_TOOLKIT_ENABLE_KOKKOS_TESTS)
    find_package(Kokkos)
    if(TENSOR_TOOLKIT_ENABLE_CUDA)
        set(Kokkos_ENABLE_CUDA ON)
    endif()
    if(TENSOR_TOOLKIT_ENABLE_HIP)
        set(Kokkos_ENABLE_HIP ON)
    endif()

    if(TENSOR_TOOLKIT_ENABLE_KOKKOS)
        target_link_libraries(tensor_toolkit
            PRIVATE INTERFACE
                Kokkos::kokkos
        )
    endif()

    if(TENSOR_TOOLKIT_ENABLE_TESTS AND TENSOR_TOOLKIT_ENABLE_KOKKOS_TESTS)
        if(NOT TENSOR_TOOLKIT_ENABLE_KOKKOS)
            target_link_libraries(ttk_unit_tests
                PRIVATE INTERFACE
                    Kokkos::kokkos
            )
        endif()
    endif()
endif()

# if(TENSOR_TOOLKIT_ENABLE_TESTS)
#     if(TENSOR_TOOLKIT_ENABLE_KOKKOS_TESTS)
#         find_package(Kokkos)
#         if(TENSOR_TOOLKIT_ENABLE_HIP)
#             set(Kokkos_ENABLE_HIP ON)
#         endif()

#         target_link_libraries(ttk_unit_tests
#             PRIVATE
#                 Kokkos::kokkos
#         )
#     endif()
# endif()
