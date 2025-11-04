from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set
from copy import deepcopy
import sys
import time  # Added timer import

# ========== TUNABLE CONSTANTS ==========
GLOBAL_TOTAL_CAP = 20000  # absolute cap on number of stored objects (safety)
PER_ITER_SEQUENT_CAP = 800  # max sequents added per iteration
PER_ITER_CONTEXT_CAP = 3000  # max contexts added per iteration
COMPOSITION_CLOSURE_ROUNDS = 3  # rounds to close composition of contexts with original subforms
WORKLIST_ITER_LIMIT = 200  # max worklist iterations during a single prove
PRACTICAL_FORMULA_CHECK_LIMIT = 200  # when pruning contexts, check at most this many formulas


# =======================================

@dataclass(eq=False)
class Node:
    val: str
    left: Optional['Node'] = None
    right: Optional['Node'] = None

    def is_hole(self) -> bool:
        return self.val == '#'

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def copy(self) -> 'Node':
        return deepcopy(self)


class SCProver:
    def __init__(self, size_limit: Optional[int] = None, trace_rules: bool = False):
        # persistent memory across prove calls
        self.formulas_table: Dict[str, Node] = {}  # formula_str -> Node (no holes)
        self.contexts_table: Dict[str, Dict] = {}  # ctx_str -> {'node': Node, 'types': set(...)}
        self.contexts_pos: Dict[str, Node] = {}  # positive contexts only
        self.contexts_neg: Dict[str, Node] = {}  # negative contexts only
        self.contexts_raw: Dict[str, Node] = {}  # raw contexts (no polarity)
        self.sequents_table: Dict[Tuple[str, str], Dict] = {}  # (L,R) -> {'L': Node, 'R': Node}

        self.allowed_formulas: Set[str] = set()  # allowed formulas (for registering sequents)

        # caches for derivability queries
        self.seq_cache: Dict[Tuple[str, str], bool] = {}  # (L,R) -> bool
        self.pos_ctx_cache: Dict[str, bool] = {}  # ctx_str -> bool (provable pos)
        self.neg_ctx_cache: Dict[str, bool] = {}  # ctx_str -> bool (provable neg)

        # sizing
        self.user_size_limit = size_limit
        self.max_size: Optional[int] = None

        # tracing
        self.trace_rules = trace_rules
        self.rule_traces: Dict[Tuple[str, str], List[str]] = {}  # optional: store small trace messages

    # ---------------- parsing / printing ----------------
    def parse_polish_tokens(self, tokens: List[str], i: int = 0) -> Tuple[Node, int]:
        if i >= len(tokens):
            raise ValueError(f"Unexpected end of tokens while parsing: tokens={tokens}")
        t = tokens[i]
        if t in {'\\', '/', '•'}:
            left, j = self.parse_polish_tokens(tokens, i + 1)
            right, k = self.parse_polish_tokens(tokens, j)
            return Node(t, left, right), k
        return Node(t), i + 1

    def parse_polish(self, s: str) -> Node:
        if isinstance(s, Node):
            return s
        tokens = s.strip().split()
        if not tokens:
            raise ValueError("Empty string")
        root, next_i = self.parse_polish_tokens(tokens, 0)
        if next_i != len(tokens):
            raise ValueError("Extra tokens after parsing: " + " ".join(tokens[next_i:]))
        return root

    def to_polish(self, node: Node) -> str:
        if node is None:
            return ""
        if node.left is None and node.right is None:
            return node.val
        return f"{node.val} {self.to_polish(node.left)} {self.to_polish(node.right)}"

    # ---------------- size computation ----------------
    def _compute_size(self, node: Node) -> int:
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return 1 + (self._compute_size(node.left) if node.left else 0) + (
            self._compute_size(node.right) if node.right else 0)

    # ---------------- subformula / subcontext generation ----------------
    def collect_subformulas(self, node: Node, out: Dict[str, Node]):
        key = self.to_polish(node)
        if key in out:
            return
        out[key] = node
        if node.left:
            self.collect_subformulas(node.left, out)
        if node.right:
            self.collect_subformulas(node.right, out)

    def _collect_paths(self, node: Node, path: List[str], out: List[List[str]]):
        # collect path to every node (including root = empty path)
        out.append(list(path))
        if node.left:
            self._collect_paths(node.left, path + ['L'], out)
        if node.right:
            self._collect_paths(node.right, path + ['R'], out)

    def generate_all_subcontexts(self, formula: Node) -> Set[str]:
        """
        Generate all contexts by replacing each subformula with a hole '#'.
        Return set of Polish strings representing contexts.
        """
        paths: List[List[str]] = []
        self._collect_paths(formula, [], paths)
        contexts = set()
        for p in paths:
            copy_root = deepcopy(formula)
            # p is path to the node to replace by hole; if p == [] that's the whole formula -> '#'
            if not p:
                contexts.add('#')
                continue
            parent = copy_root
            for step in p[:-1]:
                parent = parent.left if step == 'L' else parent.right
            if p[-1] == 'L':
                parent.left = Node('#')
            else:
                parent.right = Node('#')
            contexts.add(self.to_polish(copy_root))
        return contexts

    # ---------------- hole / composition utilities ----------------
    def _find_hole_path(self, node: Node, path: Optional[List[str]] = None) -> Optional[List[str]]:
        if path is None:
            path = []
        if node is None:
            return None
        if node.is_hole():
            return path.copy()
        if node.left:
            res = self._find_hole_path(node.left, path + ['L'])
            if res is not None:
                return res
        if node.right:
            res = self._find_hole_path(node.right, path + ['R'])
            if res is not None:
                return res
        return None

    def count_holes(self, node: Node) -> int:
        if node is None:
            return 0
        if node.is_hole():
            return 1
        return (self.count_holes(node.left) if node.left else 0) + (self.count_holes(node.right) if node.right else 0)

    def _fill_hole_rec(self, context: Node, filler: Node) -> Tuple[Node, bool]:
        """
        Replace the FIRST hole in 'context' with 'filler' (treat filler as a formula or context).
        Do not mutate inputs. Return (new_node, changed).
        """
        if context is None:
            return context, False
        if context.is_hole():
            return deepcopy(filler), True

        # try left
        if context.left:
            left_new, left_changed = self._fill_hole_rec(context.left, filler)
            if left_changed:
                new_node = Node(context.val, left_new, deepcopy(context.right) if context.right else None)
                return new_node, True

        # try right
        if context.right:
            right_new, right_changed = self._fill_hole_rec(context.right, filler)
            if right_changed:
                new_node = Node(context.val, deepcopy(context.left) if context.left else None, right_new)
                return new_node, True

        return deepcopy(context), False

    def fill_hole(self, context: Node, filler: Node) -> Node:
        new_node, changed = self._fill_hole_rec(context, filler)
        if not changed:
            raise ValueError("No hole '#' found in context: " + self.to_polish(context))
        return new_node

    def compose_contexts(self, outer: Node, inner: Node) -> Node:
        """
        Compose contexts Γ[Δ[]] by replacing the first hole in outer with the entire inner (which may contain its own hole).
        If outer is a hole, returns deepcopy(inner) (inner preserves its holes).
        IMPORTANT: preserves tree structure (non-associative).
        """
        if outer is None:
            return deepcopy(inner)
        if outer.is_hole():
            return deepcopy(inner)

        # direct child hole shortcut
        if outer.left and outer.left.is_hole():
            result = deepcopy(outer)
            result.left = deepcopy(inner)
            return result
        if outer.right and outer.right.is_hole():
            result = deepcopy(outer)
            result.right = deepcopy(inner)
            return result

        # recurse
        if outer.left:
            left_comp = self.compose_contexts(outer.left, inner)
            if self.to_polish(left_comp) != self.to_polish(outer.left):
                result = deepcopy(outer)
                result.left = left_comp
                return result
        if outer.right:
            right_comp = self.compose_contexts(outer.right, inner)
            if self.to_polish(right_comp) != self.to_polish(outer.right):
                result = deepcopy(outer)
                result.right = right_comp
                return result

        # no hole found -> return copy
        return deepcopy(outer)

    def delete_children_at_hole(self, node: Node):
        """
        After composing contexts we may have nodes with holes; delete children at hole
        so holes become leaves (hole should be leaf).
        """
        if node is None:
            return
        if node.is_hole():
            node.left = None
            node.right = None
            return
        if node.left:
            self.delete_children_at_hole(node.left)
        if node.right:
            self.delete_children_at_hole(node.right)

    # ---------------- context validators ----------------
    def is_valid_negative_context(self, node: Node) -> bool:
        """Conservative: exactly one hole and the last edge to hole is 'R' or root hole."""
        holes = self.count_holes(node)
        if holes != 1:
            return False
        p = self._find_hole_path(node)
        if p is None:
            return False
        return (len(p) == 0) or (p[-1] == 'R')

    def is_valid_positive_context(self, node: Node) -> bool:
        """Conservative: exactly one hole and the last edge to hole is 'L' or root hole."""
        holes = self.count_holes(node)
        if holes != 1:
            return False
        p = self._find_hole_path(node)
        if p is None:
            return False
        return (len(p) == 0) or (p[-1] == 'L')

    # ---------------- size / limit check ----------------
    def within_size_limit(self, node: Node, limit: Optional[int] = None) -> bool:
        if node is None:
            return True
        if limit is None:
            if self.user_size_limit is not None:
                eff = self.user_size_limit
            elif self.max_size is not None:
                eff = self.max_size
            else:
                eff = 12
        else:
            eff = limit
        return len(self.to_polish(node).split()) <= eff

    # ---------------- registration helpers ----------------
    def register_formula_node(self, node: Node) -> Tuple[str, bool]:
        """
        Register a formula node (must not contain holes). Returns (key, is_new).
        """
        pol = self.to_polish(node)
        if '#' in pol.split():
            return pol, False
        if pol not in self.formulas_table:
            # store a deepcopy to avoid external mutation surprises
            self.formulas_table[pol] = deepcopy(node)
            self.allowed_formulas.add(pol)
            return pol, True
        return pol, False

    def register_formula(self, s: str) -> str:
        node = self.parse_polish(s)
        if '#' in s.split():
            raise ValueError("Formula cannot contain hole '#': " + s)
        key, _ = self.register_formula_node(node)
        return key

    def _classify_context_types(self, node: Node) -> Set[str]:
        inferred = set()
        if self.is_valid_negative_context(node):
            inferred.add('neg')
        if self.is_valid_positive_context(node):
            inferred.add('pos')
        if not inferred:
            inferred.add('raw')
        return inferred

    def register_context(self, node: Node, types: Optional[Set[str]] = None) -> Tuple[str, bool]:
        """
        Register context node (may contain a hole). Returns (key, is_new).
        Conservative pruning: if the context has a hole, ensure plugging some known formula yields a formula
        that is within size bounds and doesn't contain a hole. This avoids contexts that cannot produce useful formulas.
        """
        # size checks
        if self.max_size is not None and self._compute_size(node) > self.max_size:
            return self.to_polish(node), False
        if not self.within_size_limit(node):
            return self.to_polish(node), False

        key = self.to_polish(node)

        # If context contains holes, check plugging some known formulas (bounded) to see if useful
        if self.count_holes(node) >= 1:
            useful = False
            # check a bounded sample of known formulas (don't iterate all if there are many)
            check_i = 0
            for fpol, fnode in list(self.formulas_table.items()):
                if check_i >= PRACTICAL_FORMULA_CHECK_LIMIT:
                    break
                check_i += 1
                try:
                    composed = self.compose_contexts(node.copy(), deepcopy(fnode))
                    self.delete_children_at_hole(composed)
                    # If the composed result is a hole-free formula and within size, it's useful
                    if '#' not in self.to_polish(composed).split() and self.within_size_limit(composed):
                        useful = True
                        break
                except Exception:
                    continue
            if not useful and len(self.formulas_table) > 0:
                # prune this context: it cannot produce a valid formula from current known formulas (conservative)
                return key, False

        is_new = False
        if key not in self.contexts_table:
            inferred = set() if types else self._classify_context_types(node)
            if types:
                inferred.update(types)
            self.contexts_table[key] = {'node': deepcopy(node), 'types': inferred}
            is_new = True
            if 'pos' in inferred:
                self.contexts_pos[key] = self.contexts_table[key]['node']
            if 'neg' in inferred:
                self.contexts_neg[key] = self.contexts_table[key]['node']
            if 'raw' in inferred:
                self.contexts_raw[key] = self.contexts_table[key]['node']
        else:
            before = len(self.contexts_table[key]['types'])
            if types:
                self.contexts_table[key]['types'].update(types)
            else:
                self.contexts_table[key]['types'].update(self._classify_context_types(node))
            if len(self.contexts_table[key]['types']) > before:
                is_new = True
            # refresh typed maps
            if 'pos' in self.contexts_table[key]['types']:
                self.contexts_pos[key] = self.contexts_table[key]['node']
            if 'neg' in self.contexts_table[key]['types']:
                self.contexts_neg[key] = self.contexts_table[key]['node']
            if 'raw' in self.contexts_table[key]['types']:
                self.contexts_raw[key] = self.contexts_table[key]['node']
        return key, is_new

    def register_sequent(self, Lnode: Node, Rnode: Node, record_rule: Optional[str] = None) -> Tuple[
        Optional[Tuple[str, str]], bool]:
        """
        Register sequent only if both sides are allowed formulas.
        Returns ((Lkey,Rkey), is_new) or (None, False) if not eligible.
        Optionally record which rule produced this sequent.
        """
        L = self.to_polish(Lnode)
        R = self.to_polish(Rnode)
        if L not in self.allowed_formulas or R not in self.allowed_formulas:
            return None, False
        key = (L, R)
        if key not in self.sequents_table:
            self.sequents_table[key] = {'L': deepcopy(Lnode), 'R': deepcopy(Rnode)}
            if record_rule and self.trace_rules:
                self.rule_traces.setdefault(key, []).append(record_rule)
            return key, True
        return key, False

    # ---------------- axioms and helper rules ----------------
    def apply_Id_and_axioms_seeding(self) -> Dict[str, int]:
        """
        Ensure identity sequents for all known formulas, and register hole context.
        Returns counts of new items.
        """
        counts = {'sequents': 0, 'contexts': 0}
        for key, node in list(self.formulas_table.items()):
            tup = (key, key)
            if tup not in self.sequents_table:
                _, added = self.register_sequent(node, node, record_rule='Id')
                if added:
                    counts['sequents'] += 1
        # register hole context '#' as both pos/neg/raw
        hole = Node('#')
        _, added = self.register_context(hole, types={'neg', 'pos', 'raw'})
        if added:
            counts['contexts'] += 1
        return counts

    # ---------------- rule applications (worklist-aware) ----------------
    def apply_connective_rules_worklist(self, new_seq_keys: Set[Tuple[str, str]], all_seq_keys: Set[Tuple[str, str]],
                                        size_limit: Optional[int], per_cap: int, counts: Dict[str, int]):
        """
        Apply \ / • using new × all and all × new, like a frontier expansion.
        We record rule names for tracing if enabled.
        """
        added = 0
        parse_cache = {}

        def p(k):
            if k not in parse_cache:
                parse_cache[k] = self.parse_polish(k)
            return parse_cache[k]

        # iterate new × all
        for (L_key, R_key) in list(new_seq_keys):
            for (C_key, D_key) in list(all_seq_keys):
                if added >= per_cap:
                    return
                try:
                    left1 = Node('\\', p(R_key), p(C_key))
                    right1 = Node('\\', p(L_key), p(D_key))
                    if self.within_size_limit(left1, size_limit) and self.within_size_limit(right1, size_limit):
                        _, a = self.register_sequent(left1, right1, record_rule='\\-rule')
                        if a:
                            counts['sequents'] += 1
                            added += 1
                except Exception:
                    pass
                if added >= per_cap:
                    return
                try:
                    left2 = Node('/', p(C_key), p(R_key))
                    right2 = Node('/', p(D_key), p(L_key))
                    if self.within_size_limit(left2, size_limit) and self.within_size_limit(right2, size_limit):
                        _, a = self.register_sequent(left2, right2, record_rule='/-rule')
                        if a:
                            counts['sequents'] += 1
                            added += 1
                except Exception:
                    pass
                if added >= per_cap:
                    return
                try:
                    left3 = Node('•', p(L_key), p(C_key))
                    right3 = Node('•', p(R_key), p(D_key))
                    if self.within_size_limit(left3, size_limit) and self.within_size_limit(right3, size_limit):
                        _, a = self.register_sequent(left3, right3, record_rule='•-rule')
                        if a:
                            counts['sequents'] += 1
                            added += 1
                except Exception:
                    pass

        # iterate all × new
        for (L_key, R_key) in list(all_seq_keys):
            for (C_key, D_key) in list(new_seq_keys):
                if added >= per_cap:
                    return
                try:
                    left1 = Node('\\', p(R_key), p(C_key))
                    right1 = Node('\\', p(L_key), p(D_key))
                    if self.within_size_limit(left1, size_limit) and self.within_size_limit(right1, size_limit):
                        _, a = self.register_sequent(left1, right1, record_rule='\\-rule')
                        if a:
                            counts['sequents'] += 1
                            added += 1
                except Exception:
                    pass
                if added >= per_cap:
                    return
                try:
                    left2 = Node('/', p(C_key), p(R_key))
                    right2 = Node('/', p(D_key), p(L_key))
                    if self.within_size_limit(left2, size_limit) and self.within_size_limit(right2, size_limit):
                        _, a = self.register_sequent(left2, right2, record_rule='/-rule')
                        if a:
                            counts['sequents'] += 1
                            added += 1
                except Exception:
                    pass
                if added >= per_cap:
                    return
                try:
                    left3 = Node('•', p(L_key), p(C_key))
                    right3 = Node('•', p(R_key), p(D_key))
                    if self.within_size_limit(left3, size_limit) and self.within_size_limit(right3, size_limit):
                        _, a = self.register_sequent(left3, right3, record_rule='•-rule')
                        if a:
                            counts['sequents'] += 1
                            added += 1
                except Exception:
                    pass

    def apply_negative_context_rules_worklist(self, new_ctx_keys: Set[str], counts: Dict[str, int], per_cap: int):
        """
        Negative context generation: conservative rule which synthesizes contexts of form • A Γ[...] or • Γ[...] A
        but we use '#' as placeholder for A when generating the context shape (the actual A is filled by Cont).
        """
        added = 0
        neg_all = list(self.contexts_neg.items())
        for gk in list(new_ctx_keys):
            if gk not in self.contexts_table:
                continue
            if 'neg' not in self.contexts_table[gk]['types']:
                continue
            gnode = self.contexts_table[gk]['node']
            for dk, dnode in neg_all:
                if added >= per_cap:
                    return
                try:
                    inner = Node('\\', deepcopy(Node('#')), deepcopy(dnode))
                    composed = self.compose_contexts(gnode.copy(), inner)
                    result = Node('•', deepcopy(Node('#')), composed)
                    self.delete_children_at_hole(result)
                    k, a = self.register_context(result, types={'neg', 'raw'})
                    if a:
                        counts['contexts'] += 1
                        added += 1
                except Exception:
                    pass
                if added >= per_cap:
                    return
                try:
                    inner2 = Node('/', deepcopy(dnode), deepcopy(Node('#')))
                    composed2 = self.compose_contexts(gnode.copy(), inner2)
                    result2 = Node('•', composed2, deepcopy(Node('#')))
                    self.delete_children_at_hole(result2)
                    k2, a2 = self.register_context(result2, types={'neg', 'raw'})
                    if a2:
                        counts['contexts'] += 1
                        added += 1
                except Exception:
                    pass

    def apply_positive_context_rules_worklist(self, new_ctx_keys: Set[str], new_seq_keys: Set[Tuple[str, str]],
                                              counts: Dict[str, int], per_cap: int):
        """
        Positive context generation. Combine pos contexts (pos x pos) and mix new sequents
        (as in your original rules) in a conservative way using the new frontier only.
        """
        added = 0
        pos_all = list(self.contexts_pos.items())
        neg_all = list(self.contexts_neg.items())

        # pos x pos (combine shapes)
        for gk in list(new_ctx_keys):
            if gk not in self.contexts_table:
                continue
            if 'pos' not in self.contexts_table[gk]['types']:
                continue
            gnode = self.contexts_table[gk]['node']
            for dk, dnode in pos_all:
                if added >= per_cap:
                    return
                try:
                    inner = Node('•', deepcopy(Node('#')), deepcopy(dnode))
                    composed, changed = self._fill_hole_rec(gnode.copy(), inner)
                    if changed:
                        result = Node('\\', deepcopy(Node('#')), composed)
                        self.delete_children_at_hole(result)
                        k, a = self.register_context(result, types={'pos', 'raw'})
                        if a:
                            counts['contexts'] += 1
                            added += 1
                except Exception:
                    pass
                if added >= per_cap:
                    return
                try:
                    inner2 = Node('•', deepcopy(dnode), deepcopy(Node('#')))
                    composed2, changed2 = self._fill_hole_rec(gnode.copy(), inner2)
                    if changed2:
                        result2 = Node('/', composed2, deepcopy(Node('#')))
                        self.delete_children_at_hole(result2)
                        k2, a2 = self.register_context(result2, types={'pos', 'raw'})
                        if a2:
                            counts['contexts'] += 1
                            added += 1
                except Exception:
                    pass

        # sequents-driven mixing: for each new sequent combine with existing neg/pos contexts
        for (X_key, Y_key) in list(new_seq_keys):
            if added >= per_cap:
                return
            B_node = self.formulas_table.get(X_key, None) or self.parse_polish(X_key)
            A_node = self.formulas_table.get(Y_key, None) or self.parse_polish(Y_key)
            for gk, gnode in neg_all:
                if added >= per_cap:
                    return
                for dk, dnode in pos_all:
                    if added >= per_cap:
                        return
                    try:
                        inner = Node('\\', deepcopy(dnode), deepcopy(B_node))
                        composed, changed = self._fill_hole_rec(gnode.copy(), inner)
                        if changed:
                            result = Node('/', deepcopy(A_node), composed)
                            self.delete_children_at_hole(result)
                            k, a = self.register_context(result, types={'pos', 'raw'})
                            if a:
                                counts['contexts'] += 1
                                added += 1
                    except Exception:
                        pass
                    if added >= per_cap:
                        return
                    try:
                        inner2 = Node('/', deepcopy(B_node), deepcopy(dnode))
                        composed2, changed2 = self._fill_hole_rec(gnode.copy(), inner2)
                        if changed2:
                            result2 = Node('\\', composed2, deepcopy(A_node))
                            self.delete_children_at_hole(result2)
                            k2, a2 = self.register_context(result2, types={'pos', 'raw'})
                            if a2:
                                counts['contexts'] += 1
                                added += 1
                    except Exception:
                        pass

    def apply_Cont_rules_worklist(self, new_ctx_keys: Set[str], new_seq_keys: Set[Tuple[str, str]],
                                  counts: Dict[str, int], per_cap: int):
        """
        Apply Cont rules to produce new sequents from (A => B) and contexts Γ[#].
        Use frontier: combine new sequents with all contexts, and all sequents with new contexts.
        """
        added = 0
        ctx_items = list(self.contexts_table.items())

        # new sequents × all contexts
        for (L_key, R_key) in list(new_seq_keys):
            if added >= per_cap:
                return
            A_node = self.formulas_table.get(L_key, None) or self.parse_polish(L_key)
            B_node = self.formulas_table.get(R_key, None) or self.parse_polish(R_key)
            for ctx_key, ctxrec in ctx_items:
                if added >= per_cap:
                    return
                ctx_node = ctxrec['node']
                types = ctxrec['types']
                if 'neg' in types and self.is_valid_negative_context(ctx_node):
                    try:
                        filled = self.compose_contexts(ctx_node.copy(), deepcopy(A_node))
                        self.delete_children_at_hole(filled)
                        if self.within_size_limit(filled):
                            _, a = self.register_sequent(filled, deepcopy(B_node), record_rule='Cont-neg')
                            if a:
                                counts['sequents'] += 1
                                added += 1
                    except Exception:
                        pass
                if added >= per_cap:
                    return
                if 'pos' in types and self.is_valid_positive_context(ctx_node):
                    try:
                        filled = self.compose_contexts(ctx_node.copy(), deepcopy(B_node))
                        self.delete_children_at_hole(filled)
                        if self.within_size_limit(filled):
                            _, a = self.register_sequent(deepcopy(A_node), filled, record_rule='Cont-pos')
                            if a:
                                counts['sequents'] += 1
                                added += 1
                    except Exception:
                        pass

        # all sequents × new contexts
        for (L_key, R_key) in list(self.sequents_table.keys()):
            if added >= per_cap:
                return
            A_node = self.formulas_table.get(L_key, None) or self.parse_polish(L_key)
            B_node = self.formulas_table.get(R_key, None) or self.parse_polish(R_key)
            for ctx_key in list(new_ctx_keys):
                if added >= per_cap:
                    return
                if ctx_key not in self.contexts_table:
                    continue
                ctx_node = self.contexts_table[ctx_key]['node']
                types = self.contexts_table[ctx_key]['types']
                if 'neg' in types and self.is_valid_negative_context(ctx_node):
                    try:
                        filled = self.compose_contexts(ctx_node.copy(), deepcopy(A_node))
                        self.delete_children_at_hole(filled)
                        if self.within_size_limit(filled):
                            _, a = self.register_sequent(filled, deepcopy(B_node), record_rule='Cont-neg')
                            if a:
                                counts['sequents'] += 1
                                added += 1
                    except Exception:
                        pass
                if added >= per_cap:
                    return
                if 'pos' in types and self.is_valid_positive_context(ctx_node):
                    try:
                        filled = self.compose_contexts(ctx_node.copy(), deepcopy(B_node))
                        self.delete_children_at_hole(filled)
                        if self.within_size_limit(filled):
                            _, a = self.register_sequent(deepcopy(A_node), filled, record_rule='Cont-pos')
                            if a:
                                counts['sequents'] += 1
                                added += 1
                    except Exception:
                        pass

    # ---------------- helpers ----------------
    def context_matches_formula(self, context_str: str, formula_str: str) -> bool:
        """
        Returns True if plugging the formula into the hole of context_str
        produces any registered formula equal to the LHS string.
        """
        context_node = self.parse_polish(context_str)
        formula_node = self.parse_polish(formula_str)
        composed = self.compose_contexts(context_node, formula_node)
        return self.to_polish(composed) in self.formulas_table

    # ---------------- main driver (per-query targeted proving with caching) ----------------
    def prove(self, lhs: str, rhs: str, verbose: bool = False, iteration_cap: int = WORKLIST_ITER_LIMIT) -> Tuple[
        bool, Dict]:
        """
        Top-level prove method. Does not clear global tables (memory persists).
        - Registers subformulas/contexts from inputs.
        - Performs a targeted worklist saturation to try to derive (lhs => rhs).
        - Uses seq_cache for memoization, with provisional False while attempting derivation.
        """
        # Start timer
        start_time = time.time()

        # parse inputs
        try:
            Lnode = self.parse_polish(lhs)
            Rnode = self.parse_polish(rhs)
        except Exception as e:
            raise ValueError(f"Parse error for inputs: {e}")

        # compute/extend size bound
        sizeA = self._compute_size(Lnode)
        sizeB = self._compute_size(Rnode)
        computed_max = max(sizeA, sizeB) * 2
        if self.max_size is None:
            self.max_size = computed_max
        else:
            # keep max_size as the maximum complexity we've ever seen (safe)
            self.max_size = max(self.max_size, computed_max)
        if self.user_size_limit is None:
            self.user_size_limit = self.max_size
        else:
            self.user_size_limit = min(self.user_size_limit, self.max_size)

        # 1) register subformulas (persistently)
        subforms: Dict[str, Node] = {}
        self.collect_subformulas(Lnode, subforms)
        self.collect_subformulas(Rnode, subforms)
        for k, n in subforms.items():
            if '#' not in k.split():
                self.register_formula_node(n)
        # ensure allowed_formulas contains these
        self.allowed_formulas.update(subforms.keys())

        # 2) register subcontexts
        ctx_strs = set()
        ctx_strs.update(self.generate_all_subcontexts(Lnode))
        ctx_strs.update(self.generate_all_subcontexts(Rnode))
        ctx_strs.add('#')
        for cstr in ctx_strs:
            try:
                node = self.parse_polish(cstr)
            except Exception:
                continue
            self.register_context(node, types=None)

        # 3) bounded composition closure: compose contexts with original subforms only (conservative)
        closure_rounds = 0
        changed_comp = True
        while changed_comp and closure_rounds < COMPOSITION_CLOSURE_ROUNDS:
            closure_rounds += 1
            changed_comp = False
            orig_form_keys = set(subforms.keys())
            ctx_items = list(self.contexts_table.items())
            for fkey in list(orig_form_keys):
                if fkey not in self.formulas_table:
                    continue
                fnode = self.formulas_tree_safe_get(fkey)
                if fnode is None:
                    continue
                for ctx_key, ctxrec in ctx_items:
                    ctx_node = ctxrec['node']
                    try:
                        composed = self.compose_contexts(ctx_node.copy(), deepcopy(fnode))
                        if '#' in self.to_polish(composed).split():
                            continue
                        if self.max_size is not None and self._compute_size(composed) > self.max_size:
                            continue
                        pol = self.to_polish(composed)
                        if pol not in self.formulas_table:
                            self.formulas_table[pol] = deepcopy(composed)
                            self.allowed_formulas.add(pol)
                            changed_comp = True
                            # safety break if global caps approaching
                            total_size = len(self.formulas_table) + len(self.contexts_table) + len(self.sequents_table)
                            if total_size > GLOBAL_TOTAL_CAP:
                                changed_comp = False
                                break
                    except Exception:
                        pass
                if not changed_comp and len(self.formulas_table) + len(self.contexts_table) > GLOBAL_TOTAL_CAP:
                    break

        # 4) seed Id and hole
        seeded = self.apply_Id_and_axioms_seeding()

        if verbose:
            print("After seeding:")
            print("  known formulas:", len(self.formulas_table))
            print("  known contexts:", len(self.contexts_table))
            print("  known sequents:", len(self.sequents_table))
            print(f"  max_size={self.max_size}, effective limit={self.user_size_limit}")

        target = (lhs, rhs)

        # quick checks: if already known
        if target in self.sequents_table:
            self.seq_cache[target] = True
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            summary = {
                'derivable': True,
                'num_sequents_registered': len(self.sequents_table),
                'num_contexts_registered': len(self.contexts_table),
                'seq_cache_size': len(self.seq_cache),
                'neg_ctx_cache_size': len(self.neg_ctx_cache),
                'pos_ctx_cache_size': len(self.pos_ctx_cache),
                'iterations': 0,
                'max_size': self.max_size,
                'effective_size_limit': self.user_size_limit,
                'proving_time': elapsed_time  # Add timing to summary
            }
            return True, summary

        # If cached answer present
        if target in self.seq_cache:
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            return self.seq_cache[target], {
                'derivable': self.seq_cache[target],
                'num_sequents_registered': len(self.sequents_table),
                'num_contexts_registered': len(self.contexts_table),
                'seq_cache_size': len(self.seq_cache),
                'neg_ctx_cache_size': len(self.neg_ctx_cache),
                'pos_ctx_cache_size': len(self.pos_ctx_cache),
                'iterations': 0,
                'max_size': self.max_size,
                'effective_size_limit': self.user_size_limit,
                'proving_time': elapsed_time  # Add timing to summary
            }

        # Set provisional cache to False to avoid infinite recursion cycles
        self.seq_cache[target] = False

        # Worklist seed: new items are ones currently in tables
        seen_sequents = set(self.sequents_table.keys())
        seen_contexts = set(self.contexts_table.keys())
        new_sequents = set(seen_sequents)
        new_contexts = set(seen_contexts)

        iterations = 0
        # main targeted worklist loop
        while iterations < iteration_cap:
            iterations += 1
            if verbose:
                print(
                    f"[prove worklist] iteration {iterations}: new_seq={len(new_sequents)}, new_ctx={len(new_contexts)}, total_seq={len(self.sequents_table)}, total_ctx={len(self.contexts_table)}")

            # early exit if target created
            if target in self.sequents_table:
                self.seq_cache[target] = True
                break

            counts = {'sequents': 0, 'contexts': 0}

            all_seq_keys = set(self.sequents_table.keys())

            # apply connective rules using frontier
            if new_sequents:
                self.apply_connective_rules_worklist(new_sequents, all_seq_keys, self.user_size_limit,
                                                     PER_ITER_SEQUENT_CAP, counts)

            # apply context generation rules
            if new_contexts:
                self.apply_positive_context_rules_worklist(new_contexts, new_sequents, counts, PER_ITER_CONTEXT_CAP)
                self.apply_negative_context_rules_worklist(new_contexts, counts, PER_ITER_CONTEXT_CAP)

            # apply Cont rules mixing frontier and all
            if new_sequents or new_contexts:
                self.apply_Cont_rules_worklist(new_contexts, new_sequents, counts, PER_ITER_SEQUENT_CAP)

            # compute new frontier by difference
            all_seq_after = set(self.sequents_table.keys())
            all_ctx_after = set(self.contexts_table.keys())

            next_new_sequents = all_seq_after - seen_sequents
            next_new_contexts = all_ctx_after - seen_contexts

            # update seen
            seen_sequents.update(next_new_sequents)
            seen_contexts.update(next_new_contexts)

            # prepare for next iteration
            new_sequents = next_new_sequents
            new_contexts = next_new_contexts

            # stop if nothing new -> fixpoint
            if not new_sequents and not new_contexts:
                if verbose:
                    print("No new items: fixpoint reached.")
                break

            # safety: global storage cap
            total_size = len(self.formulas_table) + len(self.contexts_table) + len(self.sequents_table)
            if total_size > GLOBAL_TOTAL_CAP:
                if verbose:
                    print("[prove] global total cap exceeded; stopping search early.")
                break

        # final derivability decision:
        derivable = (target in self.sequents_table) or (self.seq_cache.get(target, False) is True)

        # If not directly derivable and LHS is a context, try plugging known formulas.
        # We try only formulas we have registered (bounded) and rely on seq_cache to avoid cycles.
        if not derivable and ('#' in lhs.split()):
            # iterate formulas (bounded)
            for fpol, fnode in list(self.formulas_table.items()):
                try:
                    composed = self.compose_contexts(self.parse_polish(lhs), fnode)
                    self.delete_children_at_hole(composed)
                    composed_key = self.to_polish(composed)
                    # if (composed => rhs) is in sequents_table or cached, accept
                    if (composed_key, rhs) in self.sequents_table:
                        derivable = True
                        break
                    if (composed_key, rhs) in self.seq_cache and self.seq_cache[(composed_key, rhs)]:
                        derivable = True
                        break
                    # attempt targeted prove recursively but avoid huge recursion by using iteration limit
                    inner_ok, _ = self.prove(composed_key, rhs, verbose=False, iteration_cap=WORKLIST_ITER_LIMIT)
                    if inner_ok:
                        derivable = True
                        break
                except Exception:
                    pass

        # update cache
        self.seq_cache[target] = derivable

        elapsed_time = time.time() - start_time  # Calculate elapsed time

        summary = {
            'derivable': derivable,
            'num_sequents_registered': len(self.sequents_table),
            'num_contexts_registered': len(self.contexts_table),
            'seq_cache_size': len(self.seq_cache),
            'neg_ctx_cache_size': len(self.neg_ctx_cache),
            'pos_ctx_cache_size': len(self.pos_ctx_cache),
            'iterations': iterations,
            'max_size': self.max_size,
            'effective_size_limit': self.user_size_limit,
            'proving_time': elapsed_time  # Add timing to summary
        }
        return derivable, summary

    # small helper to safely get node from formulas_table or parse if missing
    def formulas_tree_safe_get(self, pol: str) -> Optional[Node]:
        node = self.formulas_table.get(pol, None)
        if node is not None:
            return node
        try:
            return self.parse_polish(pol)
        except Exception:
            return None


# ---------- CLI ----------
def main():
    prover = SCProver(size_limit=None, trace_rules=False)
    print("SC prover (Polish). Enter LHS and RHS as Polish strings (space-separated).")
    if len(sys.argv) >= 3:
        lhs = sys.argv[1]
        rhs = sys.argv[2]
    else:
        lhs = input("LHS > ").strip()
        rhs = input("RHS > ").strip()

    # Time the prove call
    start_time = time.time()
    ok, summary = prover.prove(lhs, rhs, verbose=True, iteration_cap=WORKLIST_ITER_LIMIT)
    elapsed_time = time.time() - start_time

    print("DERIVABLE?", ok)
    print("Summary:", summary)
    print(f"Total proving time: {elapsed_time:.4f} seconds")
    if prover.trace_rules and ok and (lhs, rhs) in prover.rule_traces:
        print("Rule trace for target:", prover.rule_traces[(lhs, rhs)])


if __name__ == "__main__":
    main()