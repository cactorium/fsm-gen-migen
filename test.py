from migen import *

from parser import fsmgen

class Foo(Module):
  def __init__(self):
    self.s = Signal()
    self.counter = Signal(8)
    x = Array(Signal(name="a") for i in range(7))
    self.test_fsm(self.counter, x)

  @fsmgen(True)
  def test_fsm(self, x, inp):
    f = Signal(2)
    while True:
      if inp == 7:
        self.s.eq(1)
      elif inp == 6:
        self.s.eq(0)
      yield
      NextValue(self.s, 1)
      NextValue(self.counter, self.counter + 1)
      yield "B"
      return f

  @fsmgen()
  def test_fsm2(self, x):
    self.s.eq(1)
    yield "A"
    NextValue(self.s, 1)
    NextValue(self.counter, self.counter + 1)
    yield B


f = Foo()
print(f.test_fsm(5, 7))
print(f.test_fsm2(5))
