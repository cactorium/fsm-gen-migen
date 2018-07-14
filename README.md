# FSM generator for Migen
**This is still very much experimental, feel free to test it out and send me any bugs.
I would recommend against using this in any production environment**

This allows you to write sequential code that is automagically converted into a FSM for Migen.
This should hopefully make it easier to write HDL modules that have complicated logic without the errorprone and tedious task of laying it out as a state machine.
Currently it only supports `while` and `if` statements, which means it should be usable for most applications where you'd want to use a FSM.

## How to use
I'm still working on bundling it up as a proper Python package, but for now you can just copy and paste `migen_fsmgen.py` into your project.

Basically, decorate any function that you want to turn into a state machine with `fsmgen()` to have the function converted into another function that returns a FSM:

```
from migen import *
from migen_fsmgen import fsmgen

# The True means that migen was imported with `from migen import *`;
# otherwise fsmgen will prefix all the appropriate structures and function calls
# with "migen"
@fsmgen(True)
def make_fsm(x, y):
  while True:
    if x == 1:
      NextValue(y, 0)
      yield
      if x == 0:
        NextValue(y, 1)
    else:
      NextValue(y, 0)
    yield

x = Signal()
y = Signal()
fsm = make_fsm(x, y)
```

The semantics are basically that control flow continues until it hits a yield statement, which will correlate with the clock edge in the HDL module.
So in the above example, the FSM will set y to 1 on the clock edge after it detects a falling edge on x.

You can name the yield points so the generate code is slightly more readable by adding a name or string after the yield:

```
# No parameter is the same as setting it to True
@fsmgen()
def make_fsm2(x, y):
  while x == 1:
    yield A
  while x == 0:
    yield B
  while x == 1:
    yield C
  NextValue(y, 1)
```

## How it works
This uses `ast` and `inspect` to extract the AST from the function, and then it processes that to model the same set of steps as a state machine.
Then it generates code to produce the state machine in migen and wraps that in a function.
If `astor` is installed, `fsmgen` also saves a copy of the Python code as an attribute named fsmgen_source for inspection.
