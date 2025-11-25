import pathlib
import sys
import unittest

import pytest

np = pytest.importorskip("numpy")

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysml import Tensor
from pysml.ddp.communication import primitives as comms


class PrimitiveTests(unittest.TestCase):
    def test_send_and_recv_between_devices(self) -> None:
        comm = comms.Communicator(["cpu:0", "cuda:0"])
        tensor = Tensor(np.ones((2,), dtype=np.float32))
        comm.send(tensor, "cuda:0", src="cpu:0")
        received = comm.recv("cuda:0", src="cpu:0")
        self.assertTrue(np.allclose(received.numpy(), tensor.numpy()))

    def test_all_reduce_sums_across_participants(self) -> None:
        comm = comms.Communicator(["cpu:0", "cpu:1"])
        first = Tensor(np.array([1.0], dtype=np.float32))
        second = Tensor(np.array([3.0], dtype=np.float32))
        partial = comm.all_reduce(first, tag="loss")
        self.assertTrue(np.allclose(partial.numpy(), first.numpy()))
        result = comm.all_reduce(second, tag="loss")
        self.assertTrue(np.allclose(result.numpy(), np.array([4.0], dtype=np.float32)))

    def test_broadcast_replicates_payload(self) -> None:
        comm = comms.Communicator(["cpu:0", "cpu:1", "cpu:2"])
        payload = Tensor(np.arange(3, dtype=np.float32))
        copies = comm.broadcast(payload, src="cpu:0")
        self.assertEqual(len(copies), 3)
        for replica in copies:
            self.assertTrue(np.allclose(replica.numpy(), payload.numpy()))


if __name__ == "__main__":
    unittest.main()
