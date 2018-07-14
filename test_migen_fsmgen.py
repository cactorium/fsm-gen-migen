from migen import *
from migen.fhdl import verilog

from migen_fsmgen import fsmgen

class Foo(Module):
  def __init__(self):
    self.s = Signal()
    self.counter = Signal(8)
    x = Array(Signal(name="a") for i in range(7))
    self.submodules += self.test_fsm(self.counter, self.counter)

  @fsmgen(True)
  def test_fsm(self, x, inp):
    f = Signal(2)
    while True:
      if inp == 1:
        self.s.eq(1)
      elif inp == 0:
        self.s.eq(0)
      yield
      NextValue(self.s, 1)
      NextValue(self.counter, self.counter + 1)
      yield "B"

  @fsmgen()
  def test_fsm2(self, x):
    self.s.eq(1)
    yield "A"
    NextValue(self.s, 1)
    NextValue(self.counter, self.counter + 1)
    yield B

  # TODO: fix this case
  @fsmgen()
  def test_fsm3(x, y):
    while x == 1:
      yield A
    NextValue(y, 0)
    while x == 0:
      yield B
    NextValue(y, 0)
    while x == 1:
      yield C
    yield C2
    NextValue(y, 1)


f = Foo()
# print(f.test_fsm(5, 7))
# print(f.test_fsm2(5))
print(f.test_fsm.fsmgen_source)
# print(verilog.convert(f))
print(f.test_fsm3.fsmgen_source)
