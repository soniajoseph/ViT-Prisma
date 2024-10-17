def get_cifar10_index_to_name(imagenet_path=None):  # TODO EdS: Combine with the imagenet version
    from vit_prisma.utils.data_utils.cifar.cifar10_dict import cifar10_classes
    return cifar10_classes


def get_mnist_index_to_name(imagenet_path=None):  # TODO EdS: Combine with the imagenet version
    from vit_prisma.utils.data_utils.mnist.mnist_dict import mnist_classes
    return mnist_classes
