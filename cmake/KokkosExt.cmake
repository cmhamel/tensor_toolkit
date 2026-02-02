if(TENSOR_TOOLKIT_ENABLE_TESTS)
    if(TENSOR_TOOLKIT_ENABLE_KOKKOS_TESTS)
        find_package(Kokkos)
        if(TENSOR_TOOLKIT_ENABLE_HIP)
            set(Kokkos_ENABLE_HIP ON)
        endif()

        target_link_libraries(ttk_unit_tests
            PRIVATE
                Kokkos::kokkos
        )
    endif()
endif()
