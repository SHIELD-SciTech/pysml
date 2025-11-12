import os
import unittest

import numpy as np

from pysml import add, distributed
from pysml.tensor import Tensor


class DistributedSingleProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        distributed.shutdown()
        for key in (
            "WORLD_SIZE",
            "RANK",
            "PMI_SIZE",
            "PMI_RANK",
            "OMPI_COMM_WORLD_SIZE",
            "OMPI_COMM_WORLD_RANK",
        ):
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        distributed.shutdown()

    def test_lazy_init_uses_local_backend_when_world_size_missing(self) -> None:
        distributed.lazy_init_from_env()
        backend = distributed.get_backend("cpu")
        self.assertEqual(backend.world_size, 1)
        self.assertEqual(backend.rank, 0)

    def test_all_reduce_noop_in_single_process(self) -> None:
        tensor = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        result = distributed.all_reduce(tensor)
        np.testing.assert_array_equal(result.data, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertIs(result, tensor)

    def test_tensor_can_route_to_communicator(self) -> None:
        distributed.lazy_init_from_env()
        tensor = Tensor(np.array([5.0], dtype=np.float32))
        communicator = tensor.communicator
        self.assertEqual(communicator.rank, 0)
        self.assertEqual(communicator.world_size, 1)


class TensorOperationsRegressionTests(unittest.TestCase):
    def test_add_matches_numpy(self) -> None:
        a = Tensor(np.array([1.0, 2.0], dtype=np.float32))
        b = Tensor(np.array([3.0, 4.0], dtype=np.float32))
        out = add(a, b)
        np.testing.assert_allclose(out.data, np.array([4.0, 6.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
