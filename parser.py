import ast
import inspect

try:
  import astor
except ImportError:
  astor = None

next_id = 0
def generate_id():
  global next_id
  ret = next_id
  next_id = next_id + 1
  return ret

# five phase parse;
# - the first phase traverses the original AST tree and extracts the 
# initialization code, while loops, and if statements,
# - the second phase breaks the while loops down into if statements,
# - and the third phase merges together expressions that are combined across
# control flow boundaries
# - fourth removes redundant states
# - fifth names unnamed states
# NOTE: Python doesn't have support for case/switch statements, so the best you
# can do is probably use a bunch of if-elif-else statements, or convert the case statement
# into a bunch of flag signals and if statements
# TODO: maybe detect if an if-elif-else chain can be converted into a case statement,
# and convert it during the second pass, will probably need a new type

# follow next_node to follow AST order, follow gotos to follow control flow order
class Node:
  def __init__(self):
    self.next_node = None
    self.mapped_state = None

# a node whose type hasn't been decided yet; there'll be a lot of these, mostly
# for forward referencing to the next node
class LazyNode(Node):
  def __init__(self):
    Node.__init__(self)
    self.value = None

class BasicBlock(Node):
  def __init__(self, lineno, col_offset):
    Node.__init__(self)
    self.stmts = list()
    self.id = generate_id()
    self.lineno = lineno
    self.col_offset = col_offset

class Yield(Node):
  def __init__(self, lineno, col_offset, name=None):
    Node.__init__(self)
    self.name = None
    self.lineno = lineno
    self.col_offset = col_offset

# Goto and subclasses don't need a line number or column offset because they're
# not included in the final AST
class Goto(Node):
  def __init__(self, target, name=None):
    Node.__init__(self)
    self.target = target
    self.next_node = target

class WhileEnd(Goto):
  def __init__(self, next_node, name=None):
    Goto.__init__(self, next_node, name)

class IfEnd(Goto):
  def __init__(self, next_node, name=None):
    Goto.__init__(self, next_node, name)

class Continue(Goto):
  def __init__(self, next_node, name=None):
    Goto.__init__(self, next_node, name)

class Break(Goto):
  def __init__(self, next_node, name=None):
    Goto.__init__(self, next_node, name)

class TerminatingNode(Node):
  def __init__(self):
    Node.__init__(self)

class If(Node):
  def __init__(self, expr, then, els=None, lineno=None, col_offset=None):
    Node.__init__(self)
    self.expr = expr
    self.then = then
    self.els = els
    self.lineno = lineno
    self.col_offset = col_offset
    self.has_yields = False
    self.else_has_yields = False
    self.has_continue = False
    self.has_break = False
    self.else_has_continue = False
    self.else_has_break = False

# basically the control flow will follow into test_block
class While(Node):
  def __init__(self, expr, body, test_block, lineno=None, col_offset=None):
    Node.__init__(self)
    self.expr = expr
    self.body = body
    self.test_block = test_block
    self.lineno = lineno
    self.col_offset = col_offset
    self.has_yields = False

class ParsePass:
  def __init__(self):
    pass

  def parse_if(self, n, filename, firstline, break_target, continue_target, next_block):
    # print(firstline)
    condition = n.test

    end_block = IfEnd(next_block)
    body, has_yields, has_continues, has_breaks = self.parse_body(n.body, None, filename, firstline, break_target, continue_target, end_block)
    else_block = None
    else_has_yields = False
    else_has_continues = False
    else_has_breaks = False
    if n.orelse is not None:
      else_block, else_has_yields, else_has_continues, else_has_breaks = self.parse_body(n.orelse, None, filename, firstline, break_target, continue_target, end_block)
    if_block = If(condition, body, else_block, n.lineno - 2 + firstline, n.col_offset)
    if_block.next_node = next_block
    if_block.has_yields = has_yields
    if_block.else_has_yields = else_has_yields
    if_block.has_continue = has_continues
    if_block.has_break = has_breaks
    if_block.else_has_continue = else_has_continues
    if_block.else_has_break = else_has_breaks

    return if_block

  def parse_while(self, n, filename, firstline, next_block):
    condition = n.test

    test_goto = Goto(None)
    exit_goto = Goto(next_block)
    while_end_block = WhileEnd(test_goto)

    test_block = If(condition, test_goto, exit_goto)
    test_block.has_yields = True   # mark as true; we can assume it's true because we'll error out otherwise soon
    test_block.else_has_yields = True
    test_block.lineno = n.lineno - 2 + firstline
    test_block.col_offset = n.col_offset

    body, has_yields, has_continues, has_breaks = self.parse_body(n.body, None, filename, firstline, next_block, test_block, while_end_block)

    test_block.has_continue = has_continues
    test_block.has_break = has_breaks
    test_block.next_node = body

    test_goto.target = body
    test_goto.next_node = body

    # while loop without yields will cause an infinite loop during the merge step 
    # (since the terminating condition is either hitting a yield or the terminator), so it'd be good to avoid that
    if not has_yields:
      raise Error("[ERR] while loop without yield unsupported, found on {}:{}: {}".format(filename, firstline + n.lineno - 2, ast.dump(n)))

    # TODO: make sure that there's a yield before any continue in the loop; otherwise it could lead to the same problem
    # as a while without any yields
    # this is currently done in the final phase, which makes it a little less accurate

    while_block = While(condition, body, test_block, n.lineno - 2 + firstline, n.col_offset)
    while_block.next_node = next_block
    while_block.has_yields = has_yields

    return while_block

  def parse_body(self, body, init, filename, firstline, break_target, continue_target, next_block):
    cur_block = BasicBlock(firstline, -1)
    if len(body) > 0:
      cur_block.lineno = body[0].lineno - 2 + firstline
      cur_block.col_offset = body[0].col_offset
    first_block = cur_block
    has_yields = False
    has_breaks = False
    has_continue = False
    for n in body:
      if isinstance(n, ast.Expr):
        value = n.value
        if isinstance(value, ast.Yield):
          has_yields = True

          yield_block = Yield(value.lineno - 2 + firstline, value.col_offset)
          # TODO: grab the name from the AST if it's a string literal or a variable name
          cur_block.next_node = yield_block
          cur_block = BasicBlock(value.lineno - 2 + firstline, value.col_offset)
          yield_block.next_node = cur_block
        else:
          cur_block.stmts.append(n)
      elif isinstance(n, ast.Assign):
        # assignments shouldn't exist in the logic; it should all be a.eq(b)
        # or NextValue(a, b)
        # so I can hoist all of them out to the top of the code
        if init is not None:
          init.append(n)
        else:
          raise Error("[ERR] nonsignal assignments in if/while blocks are unsupported, found on {}:{}: {}".format(filename, firstline + n.lineno - 2, ast.dump(n)))
      elif isinstance(n, ast.If):
        if_block = self.parse_if(n, filename, firstline, break_target, continue_target, next_block)
        has_yields = has_yields or if_block.has_yields
        cur_block.next_node = if_block
        cur_block = BasicBlock(n.lineno - 2 + firstline, n.col_offset)
        if_block.next_node = cur_block
      elif isinstance(n, ast.Break):
        has_breaks = True
        break_block = Break(break_target)

        cur_block.next_node = break_block
        cur_block = break_block
        # skip the rest of the body
        break
      elif isinstance(n, ast.Continue):
        has_continue = True
        continue_block = Continue(continue_target)

        cur_block.next_node = continue_block
        cur_block = continue_block
        break
      elif isinstance(n, ast.While):
        new_block = BasicBlock(n.lineno - 2 + firstline, n.col_offset)

        while_block = self.parse_while(n, filename, firstline, new_block)
        has_yields = has_yields or while_block.has_yields

        cur_block.next_node = while_block
        cur_block = new_block
      else:
        print("[WARN] dunno how to deal with {} in {}:{}".format(ast.dump(n), filename, firstline + n.lineno - 2))
      # print(cur_block, cur_block.next_node)
    # the next_node is already set for cur_block if it was a break or continue
    if not has_breaks and not has_continue:
      cur_block.next_node = next_block
    return first_block, has_yields, has_continue, has_breaks
  
  # returns first node of the control flow
  def parse_func(self, f, func_decl):
    first_line = f.__code__.co_firstlineno
    init = list()
    first_block, has_yields, has_continues, has_breaks = self.parse_body(
        func_decl.body, init, inspect.getsourcefile(f), f.__code__.co_firstlineno, None, None, TerminatingNode())
    return first_block, init

def isnotterminating(cur_node):
  return (not isinstance(cur_node, TerminatingNode) and
      not isinstance(cur_node, IfEnd) and
      not isinstance(cur_node, WhileEnd) and
      not isinstance(cur_node, Continue) and
      not isinstance(cur_node, Break) and 
      not isinstance(cur_node, Goto))


class SimplifyPass:
  def __init__(self):
    pass

  # optimize out True/False expr values
  def simplify_if(self, prev_node, if_node):
    # print("start if simplify")
    if_node.then = self.simplify(if_node.then)
    if if_node.els is not None:
      if_node.els = self.simplify(if_node.els)
    if isinstance(if_node.expr, ast.NameConstant):
      if if_node.expr.value is True:
        prev_node.next_block = if_node.then
      elif if_node.expr.value is False:
        if if_node.els is not None:
          prev_node.next_block = if_node.els
        else:
          prev_node.next_block = if_node.next_block
    # print("end if simplify")
 
  # run the simplification starting with the test block
  # TODO: figure out how to optimize the if statement in the while loop
  # without destroying its structure; currently allowing the simplify_if
  # method to run over the test block could replace the while loop with 
  # a goto, which would break state generation because the states within
  # the loop won't be reachable because the node traversal stops when it
  # reaches any goto
  # this means that for now there'll be a few if(true) or if(false) still
  # in the generated code
  def simplify_while(self, prev_node, while_node):
    #print("start while simplify")
    while_node.body = self.simplify(while_node.body)
    #print("end while simplify")
  
  def simplify(self, start_node):
    # point at the start node with a goto so that there isn't any special case code
    # for the first node
    goto_node = Goto(start_node)
    prev_node = goto_node
    cur_node = start_node
    # terminate on any node that escapes the current scope to ensure
    # termination
    while isnotterminating(cur_node):
      assert cur_node is not None
      #print(cur_node)
      if isinstance(cur_node, While):
        self.simplify_while(prev_node, cur_node)
      elif isinstance(cur_node, If):
        self.simplify_if(prev_node, cur_node)
      # remove extra indirection
      elif isinstance(cur_node, LazyNode):
        prev_node.next_node = cur_node.value
        cur_node = cur_node.value
        continue

      prev_node, cur_node = cur_node, cur_node.next_node
    return goto_node.next_node

class State:
  def __init__(self, stmts, lineno, col_offset, name=None):
    # NOTE: stmts may contain both AST nodes and StateIf and StateTransition objects
    self.stmts = stmts
    self.name = name
    self.lineno = lineno
    self.col_offset = col_offset
  def dump(self):
    return self.name + ": " + str(self.stmts)

class StateIf:
  def __init__(self, expr, then_stmts, els_stmts, lineno, col_offset):
    self.expr = expr
    self.then = then_stmts
    self.els = els_stmts
    self.lineno = lineno
    self.col_offset = col_offset

class StateTransition:
  def __init__(self, node, name, lineno, col_offset):
    self.target = node
    self.name = name # store the name taken from the yield statement; used for state naming
    self.lineno = lineno
    self.col_offset = col_offset


class MergePass:
  def __init__(self):
    self.names = dict()
    self.names["s"] = list()

  def visit_body(self, states_out, cur_node, firstline, start_state=True):
    # pass through all the nodes in AST order, starting a state after any
    # yield statement
    while isnotterminating(cur_node):
      if start_state:
        new_state = self.generate_state(cur_node, firstline)
        cur_node.mapped_state = new_state
        states_out.append(new_state)
        start_state = False
      # start a new state at the statement after each yield statement
      # all recursive visits should have start_state equal to false; there'll always be
      # a basic block (even if it's empty) at the starts and ends of a while or if statement
      if isinstance(cur_node, If):
        assert start_state == False
        self.visit_body(states_out, cur_node.then, firstline, start_state=False)
        if cur_node.els is not None:
          self.visit_body(states_out, cur_node.els, firstline, start_state=False)
      elif isinstance(cur_node, While):
        assert start_state == False
        self.visit_body(states_out, cur_node.test_block, firstline, start_state=False)
      elif isinstance(cur_node, Yield):
        start_state = True
      cur_node = cur_node.next_node

  # returns if_stmt, next_node, is_terminating
  def gather_if_stmt(self, cur_node):
    # special case for purely combinatorial if statements, where there is no yield
    # a combinatorial if statement also cannot have any continues or breaks; those both
    # will eventually hit a yield
    if (not cur_node.has_yields and not cur_node.else_has_yields and not cur_node.has_continue and not cur_node.has_break
        and not cur_node.else_has_continue and not cur_node.else_has_break):
      then_branch = self.gather_if_block(cur_node.then)
      else_branch = None

      if cur_node.els is not None:
        else_branch = self.gather_if_block(cur_node.els)
      if_stmt = StateIf(cur_node.expr, then_branch, else_branch, cur_node.lineno, cur_node.col_offset)
      return if_stmt, cur_node.next_node, False
    else:
      # there's at least one yield
      # store all the statements within the if block so that
      # there's no way for combinatorial logic to leak in the yielded branch
      # for example, if there was something like:
      # if a > 3:
      #   b = 2
      #   yield
      #   b = 4
      # else:
      #   b = 1
      # c = 7
      # yield
      # 
      # would produce (if statement gathering wasn't hoisted into the if):
      # if a > 3:
      #   b = 2
      #   NextState("s2")
      # else:
      #   b = 1
      # c = 7
      # NextState("s3")
      # so c would be set to 7 and the next state would be set to "s3" instead of "s2"
      # 
      # hoisting fixes this by converting it instead into:
      # if a > 3:
      #   b = 2
      #   NextState("s2")
      # else:
      #   b = 1
      #   c = 7
      #   NextState("s3")
      then_branch = self.gather_stmts(cur_node.then)
      else_branch = None
      if cur_node.els is not None:
        else_branch = self.gather_stmts(cur_node.els)
      else:
        else_branch = self.gather_stmts(cur_node.next_node)
      if_stmt = StateIf(cur_node.expr, then_branch, else_branch, cur_node.lineno, cur_node.col_offset)
      return if_stmt, None, True

  def gather_if_block(self, start_node):
    stmts = []
    cur_node = start_node
    # append statements until reaching an IfEnd(...)
    while not isinstance(cur_node, IfEnd):
      #print(cur_node, cur_node.next_node)
      if isinstance(cur_node, BasicBlock):
        stmts.extend(cur_node.stmts)
      elif isinstance(cur_node, If):
        if_stmt, next_node, is_terminating = self.gather_if_stmt(cur_node)
        # there shouldn't be any yield statements
        assert not is_terminating
        stmts.append(if_stmt)
        cur_node = next_node
      elif isinstance(cur_node, While):
        # this region has no yields, so this while has no yields, which
        # means it's bad
        raise Exception("found yieldless while block at {}:{}", cur_node.lineno, cur_node.col_offset)
      cur_node = cur_node.next_node
    return stmts

  def gather_stmts(self, start_node):
    # append statments until reaching a yield at all control flow points
    stmts = []
    block_ids = set()
    done = False

    cur_node = start_node
    while not done:
      #print("loop: ", cur_node, stmts)
      assert cur_node is not None
      if isinstance(cur_node, BasicBlock):
        # check for loops; we shouldn't be able to add the same basic block twice to the
        # same state
        if cur_node.id in block_ids:
          raise Exception(
              "combinatorial loop found near {}:{}; please make sure there are yields in while statements!".format(
                cur_node.lineno, cur_node.col_offset))

        block_ids.add(cur_node.id)
        stmts.extend(cur_node.stmts)
        cur_node = cur_node.next_node
      elif isinstance(cur_node, If):
        if_stmt, next_node, is_terminating = self.gather_if_stmt(cur_node)
        stmts.append(if_stmt)
        done = is_terminating
        cur_node = cur_node.next_node
      elif isinstance(cur_node, While):
        # while statement's already been desugared into an if statement with gotos,
        # so just follow that path
        cur_node = cur_node.test_block
      elif isinstance(cur_node, Goto):
        # follow any gotos
        cur_node = cur_node.next_node
      elif isinstance(cur_node, Yield):
        stmts.append(StateTransition(cur_node.next_node, cur_node.name, cur_node.lineno, cur_node.col_offset))
        done = True
      elif isinstance(cur_node, TerminatingNode):
        done = True
      elif isinstance(cur_node, LazyNode):
        cur_node = cur_node.value

    #print(stmts)
    return stmts
  
  def generate_state(self, start_node, firstline):
    stmts = self.gather_stmts(start_node)
    #print("state ", stmts)
    lineno = -1
    col_offset = -1
    name = None
    if len(stmts) > 0:
      lineno = stmts[0].lineno - 2 + firstline
      col_offset = stmts[0].col_offset
      if isinstance(stmts[-1], StateTransition):
        name = stmts[-1].name
    new_state = State(stmts, lineno, col_offset, name)
    start_node.mapped_state = new_state
    return new_state

  def resolve_links(self, stmts):
    # fix state transitions in any statements in this state
    # state generation isn't ordered in a way that guarantees yields point to
    # already created states, so gather_stmts() always stores the matching
    # CFG node when, and generate_states() stores the state in that node's mapped_state
    # whenever it generates the right state
    # this just passes through any stmts and checks for yields to replace 
    # the links to CFG nodes with links to other states
    # this also acts as a validation point to ensure all the states were actually
    # generated, and copies names from terminating yield statements to name
    # the appropriate states
    for stmt in stmts:
      if isinstance(stmt, StateTransition):
        assert stmt.target
        assert stmt.target.mapped_state is not None
        stmt.target = stmt.target.mapped_state
      elif isinstance(stmt, StateIf):
        self.resolve_links(stmt.then)
        if stmt.els is not None:
          self.resolve_links(stmt.els)

  def name_state(self, state):
    # name a state if it doesn't already have a name
    if state.name is None:
      numeric_names = [int(name) for name in self.names["s"] if name.isdigit()]
      numeric_names.append(-1)
      last_num = max(numeric_names)
      state.name = "s_" + str(last_num + 1)
      self.names["s"].append(str(last_num + 1))
    else:
      # check to see if the name's already been taken by some other
      # state, and add a number to the end of it if it as
      parts = state.name.split("_")
      prefix = "_".join(parts[:-1])
      if prefix in self.names:
        if parts[-1] in self.names[prefix]:
          if parts[-1].isdigit():
            state.name = prefix + "_" + str(int(parts[-1]) + 1)
          else:
            state.name = state.name + "_0"
          parts = state.name.split("_")
          prefix = "_".join(parts[:-1])

      if prefix not in self.names:
        self.names[prefix] = list()
      self.names[prefix].append(parts[-1])

  def generate_states(self, start_node, firstline):
    states = []
    self.visit_body(states, start_node, firstline)
    for state in states:
      self.resolve_links(state.stmts)
    for state in states:
      self.name_state(state)
    return states

class AstGenerator:
  def __init__(self, wildcard_import=True):
    self.wildcard_import = wildcard_import

  def flatten_list(self, ll):
    ret = []
    for l in ll:
      if isinstance(l, list):
        ret.extend(self.flatten_list(l))
      else:
        ret.append(l)
    return ret

  def generate_if_stmt(self, e, wrap_expr):
    # TODO: optimize if e.expr is True or False; inject in the appropriate
    # code branch instead of an if statement
    # NOTE: will need to change return type and adjust generate_expr appropriately
    test = e.expr

    then_stmts = self.flatten_list([self.generate_expr(s, False) for s in e.then])
    els_stmts = []

    if e.els is not None and len(e.els) > 0:
      els_stmts = self.flatten_list([self.generate_expr(s, False) for s in e.els])

    if isinstance(e.expr, ast.NameConstant):
      if e.expr.value is True:
        return then_stmts
      elif e.expr.value is False:
        return els_stmts

    if_func = None
    if self.wildcard_import:
      if_func = ast.Name(id="If", ctx=ast.Load(), lineno=e.lineno, col_offset=e.col_offset)
    else:
      if_func = ast.Attribute(
            value=ast.Name(id="migen", ctx=ast.Load(), lineno=e.lineno, col_offset=e.col_offset),
            attr="If",
            ctx=ast.Load(),
            lineno=e.lineno,
            col_offset=e.col_offset)

    if_stmt = ast.Call(
          func=if_func,
          args=[test] + then_stmts,
          keywords=[],
          lineno=e.lineno,
          col_offset=e.col_offset)
    if e.els is None or len(e.els) == 0:
      if wrap_expr:
        return ast.Expr(value=if_stmt, lineno=e.lineno, col_offset=e.col_offset)
      else:
        return if_stmt
    else:
      with_els = ast.Call(
          func=ast.Attribute(
            value=if_stmt,
            attr="Else",
            ctx=ast.Load(),
            lineno=e.lineno,
            col_offset=e.col_offset),
          args=els_stmts,
          keywords=[],
          lineno=e.lineno,
          col_offset=e.col_offset)
      if wrap_expr:
        return ast.Expr(value=with_els, lineno=e.lineno, col_offset=e.col_offset)
      else:
        return with_els

  def generate_expr(self, e, wrap_expr=False):
    if isinstance(e, StateIf):
      return self.generate_if_stmt(e, wrap_expr)
    elif isinstance(e, StateTransition):
      newstate_func = None
      if self.wildcard_import:
        newstate_func = ast.Name(id="NextState", ctx=ast.Load(), lineno=e.lineno, col_offset=e.col_offset)
      else:
        newstate_func = ast.Attribute(
          value=ast.Name(id="migen", ctx=ast.Load(), lineno=e.lineno, col_offset=e.col_offset),
          attr="NextState",
          ctx=ast.Load(),
          lineno=e.lineno,
          col_offset=e.col_offset)

      call_expr = ast.Call(
              func=newstate_func,
              args=[ast.Str(s=e.target.name, ctx=ast.Load(), lineno=e.lineno, col_offset=e.col_offset)],
              keywords=[],
              lineno=e.lineno,
              col_offset=e.col_offset)
      if wrap_expr:
        return ast.Expr(
            value=call_expr,
            lineno=e.lineno,
            col_offset=e.col_offset)
      else:
        return call_expr
    else:
      if not wrap_expr and isinstance(e, ast.Expr):
        return e.value
      else:
        return e

  def generate_ast(self, state):
    state_name = ast.Str(s=state.name, ctx=ast.Load(), lineno=state.lineno, col_offset=state.col_offset)
    stmts = self.flatten_list([self.generate_expr(s, False) for s in state.stmts])
    return ast.Expr(
        value=ast.Call(
          func=ast.Attribute(
            value=ast.Name(id="fsm", ctx=ast.Load(), lineno=state.lineno, col_offset=state.col_offset),
            attr="act",
            ctx=ast.Load(),
            lineno=state.lineno,
            col_offset=state.col_offset),
          args=[state_name] + stmts,
          keywords=[],
          lineno=state.lineno,
          col_offset=state.col_offset),
        lineno=state.lineno,
        col_offset=state.col_offset)

def fsmgen(wildcard_import=True):
  def decorator(f):
    tree = ast.parse("with 0:\n" + inspect.getsource(f))
    func_decl = tree.body[0].body[0]
    #print(ast.dump(func_decl))
    #print()
    #print()
    name = f.__name__

    start_block, init = ParsePass().parse_func(f, func_decl)
    start_block = SimplifyPass().simplify(start_block)
    firstline = f.__code__.co_firstlineno
    states = MergePass().generate_states(start_block, firstline)

    ast_stmts = []
    for state in states:
      ast_stmts.append(AstGenerator(wildcard_import).generate_ast(state))

    # print([state.dump() for state in states])

    fsm_name = None
    if wildcard_import:
      fsm_name = ast.Name(id="FSM", ctx=ast.Load(), lineno=firstline, col_offset=0)
    else:
      fsm_name = ast.Attribute(
              value=ast.Name(id="migen", ctx=ast.Load(), lineno=firstline, col_offset=0),
              attr="FSM",
              ctx=ast.Load(),
              lineno=firstline,
              col_offset=0)

    prologue = [
        ast.Assign(
          targets=[ast.Name(id="fsm", ctx=ast.Store(), lineno=firstline, col_offset=0)],
          value=ast.Call(
            func=fsm_name,
            args=[],
            keywords=[
              ast.keyword(
                arg="reset_state",
                value=ast.Str(states[0].name, lineno=firstline, col_offset=0),
                lineno=firstline,
                col_offset=0)
              ],
            lineno=firstline,
            col_offset=0),
          lineno=firstline,
          col_offset=0)
        ]
    epilogue = [
        ast.Return(
          value=ast.Name(id="fsm", ctx=ast.Load(), lineno=firstline, col_offset=0),
          lineno=firstline,
          col_offset=0)
        ]
    new_body = prologue + init + ast_stmts + epilogue

    new_func_decl = ast.Module(body=[
      ast.FunctionDef(
        name=func_decl.name,
        args=func_decl.args,
        body=new_body,
        decorator_list=func_decl.decorator_list[:-1],    # remove this decorator from the list
        returns=func_decl.returns,
        lineno=firstline,
        col_offset=0)
      ])
    # print(ast.dump(new_func_decl))
    if astor is not None:
      f.fsmgen_source = astor.to_source(new_func_decl)
    p = compile(new_func_decl, "<ast>", mode="exec")
    exec(p)
    f.__code__ = locals()[func_decl.name].__code__
    return f
  return decorator
