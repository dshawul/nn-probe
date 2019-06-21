load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")

tf_cc_shared_object(
    name = "libnnprobe.so",
    srcs = [
	"my_types.h",
	"nnprobe.h",
        "nnprobe.cpp"
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow"
    ],
    defines = [ "TENSORFLOW", "PARALLEL", "USE_SPINLOCK", "ARC_64BIT", "HAS_POPCNT", "HAS_PREFETCH" ]
)
