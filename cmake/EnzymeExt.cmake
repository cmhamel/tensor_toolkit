if(TENSOR_TOOLKIT_ENABLE_ENZYME)
    find_package(Enzyme REQUIRED CONFIG)
    message(STATUS "ENZYME_Dir=${Enzyme_DIR}")
    get_filename_component(PARENT_DIR1 "${Enzyme_DIR}" DIRECTORY)
    get_filename_component(PARENT_DIR2 "${PARENT_DIR1}" DIRECTORY)
    get_filename_component(PARENT_DIR3 "${PARENT_DIR2}" DIRECTORY)
    set(ENZYME_PLUGIN ${PARENT_DIR3}/lib/LLDEnzyme-18.so)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       lld)  # <-- this is the key

    target_compile_definitions(tensor_toolkit
        PUBLIC INTERFACE
            TENSOR_TOOLKIT_ENABLE_ENZYME
    )
    target_include_directories(tensor_toolkit
        INTERFACE
            $<BUILD_INTERFACE:${PARENT_DIR3}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(tensor_toolkit
        PUBLIC INTERFACE
            LLDEnzymeFlags
    )
    if(TENSOR_TOOLKIT_ENABLE_TESTS)
        target_compile_definitions(tensor_toolkit
            PUBLIC INTERFACE
                TENSOR_TOOLKIT_ENABLE_ENZYME
        )
        target_compile_options(ttk_unit_tests 
            PRIVATE
                -fplugin=${ENZYME_PLUGIN}
                -flto
        )
        target_link_options(ttk_unit_tests 
            PRIVATE 
            -fuse-ld=lld
            -Wl,--load-pass-plugin=${ENZYME_PLUGIN}
        )
    endif()
endif()
