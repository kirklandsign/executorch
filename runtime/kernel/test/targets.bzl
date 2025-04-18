load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "operator_registry_test",
        srcs = [
            "operator_registry_test.cpp",
        ],
        headers = ["test_util.h"],
        deps = [
            "//executorch/runtime/kernel:operator_registry",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
    )

    runtime.cxx_test(
        name = "operator_registry_max_kernel_num_test",
        srcs = [
            "operator_registry_max_kernel_num_test.cpp",
        ],
        deps = [
            "//executorch/runtime/kernel:operator_registry_MAX_NUM_KERNELS_TEST_ONLY",
            "//executorch/runtime/kernel:kernel_runtime_context",
        ],
    )

    et_operator_library(
        name = "executorch_all_ops",
        include_all_operators = True,
        define_static_targets = True,
    )

    runtime.export_file(
        name = "functions.yaml",
    )

    executorch_generated_lib(
        name = "specialized_kernel_generated_lib",
        deps = [
            ":executorch_all_ops",
            "//executorch/kernels/portable:operators",
        ],
        functions_yaml_target = ":functions.yaml",
        visibility = [
            "//executorch/...",
        ],
    )

    runtime.cxx_test(
        name = "kernel_double_registration_test",
        srcs = [
            "kernel_double_registration_test.cpp",
        ],
        deps = [
            "//executorch/runtime/kernel:operator_registry",
            ":specialized_kernel_generated_lib",
        ],
    )

    executorch_generated_lib(
        name = "test_manual_registration_lib",
        deps = [
            ":executorch_all_ops",
            "//executorch/kernels/portable:operators",
        ],
        functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
        manual_registration = True,
        visibility = [
            "//executorch/...",
        ],
    )

    runtime.cxx_test(
        name = "test_kernel_manual_registration",
        srcs = [
            "test_kernel_manual_registration.cpp",
        ],
        deps = [
            "//executorch/runtime/kernel:operator_registry",
            ":test_manual_registration_lib",
        ],
    )

    for aten_mode in get_aten_mode_options():
        aten_suffix = "_aten" if aten_mode else ""

        runtime.cxx_test(
            name = "kernel_runtime_context_test" + aten_suffix,
            srcs = [
                "kernel_runtime_context_test.cpp",
            ],
            deps = [
                "//executorch/runtime/kernel:kernel_runtime_context" + aten_suffix,
                ":specialized_kernel_generated_lib",
            ],
        )

        if aten_mode:
            # Make sure we can depend on both generated_lib and generated_lib_aten
            # in the same binary.
            runtime.cxx_test(
                name = "test_generated_lib_and_aten",
                srcs = ["test_generated_lib_and_aten.cpp"],
                deps = [
                    "//executorch/kernels/portable:generated_lib",
                    "//executorch/kernels/portable:generated_lib_aten",
                    "//executorch/runtime/kernel:operator_registry_aten",
                ],
            )
