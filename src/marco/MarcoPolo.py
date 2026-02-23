import os
import queue
import threading
from collections import deque
from typing import List, Optional, Set


class MarcoPolo(object):
    def __init__(self, csolver, msolver, stats, config, pipe=None):
        self.subs = csolver
        self.map = msolver
        self.seeds = SeedManager(msolver, stats, config)
        self.stats = stats
        self.config = config
        self.bias_high = self.config['bias'] == 'MUSes'  # used frequently
        self.n = self.map.n   # number of constraints
        self.got_top = False  # track whether we've explored the complete set (top of the lattice)

        # Adaptive mode toggles MUS/MCS bias online based on recent output ratio.
        self.adaptive = bool(self.config.get('adaptive', False))
        self.adaptive_window = max(1, int(self.config.get('adaptive_window', 32)))
        self.adaptive_min_outputs = max(1, int(self.config.get('adaptive_min_outputs', 16)))
        self.adaptive_target_mus_ratio = float(self.config.get('adaptive_target_mus_ratio', 0.50))
        self.adaptive_hysteresis = max(0.0, float(self.config.get('adaptive_hysteresis', 0.15)))
        self._recent_outputs = deque(maxlen=self.adaptive_window)

        # Dual-side feedback: learn map clauses from intermediate SAT/UNSAT seeds.
        self.feedback_enabled = bool(self.config.get('feedback_enabled', False))
        self.feedback_sat_clause_max = int(self.config.get('feedback_sat_clause_max', 12))
        self.feedback_unsat_clause_max = int(self.config.get('feedback_unsat_clause_max', 12))
        self.feedback_max_clauses = int(self.config.get('feedback_max_clauses', 2000))
        self.feedback_clause_count = 0
        # Antichain maintenance:
        # - keep maximal SAT sets (strongest block_down clauses)
        # - keep minimal UNSAT sets (strongest block_up clauses)
        self._feedback_sat_maximal: List[Set[int]] = []
        self._feedback_unsat_minimal: List[Set[int]] = []
        self._feedback_sat_keys = set()
        self._feedback_unsat_keys = set()

        # Core-guided smart shrinking.
        self.smart_core = bool(self.config.get('smart_core', False))
        self.core_handoff = int(self.config.get('core_handoff', -1))
        if self.core_handoff <= 0:
            self.core_handoff = max(8, self.n // 3)
        self.core_handoff = max(1, self.core_handoff)
        self.core_base_ratio = max(1, int(self.config.get('core_base_ratio', 2)))
        self.core_backoff_cap = max(0, int(self.config.get('core_backoff_cap', 8)))
        self.core_certify = bool(self.config.get('core_certify', True))
        self.core_isect: Optional[Set[int]] = None
        self.known_muses: List[Set[int]] = []
        self.known_mus_keys = set()
        self._deletion_order = self._build_deletion_order()
        self.total_outputs = 0
        self.mus_outputs = 0

        # Portfolio policy: start cheap (no smart-core shrink), then enable smart-core
        # after enough evidence (MUS and/or output count) so early timeout risk is lower.
        self.portfolio = bool(self.config.get('portfolio', False))
        self.portfolio_smart_after_mus = max(0, int(self.config.get('portfolio_smart_after_mus', 1)))
        self.portfolio_smart_after_outputs = max(0, int(self.config.get('portfolio_smart_after_outputs', 0)))
        self.smart_core_target = bool(self.smart_core)
        if self.portfolio and self.smart_core_target:
            self.smart_core_active = False
            self.stats.increment_counter("portfolio.smart.delayed")
        else:
            self.smart_core_active = bool(self.smart_core_target)

        self.pipe = pipe
        # if a pipe is provided, use it to receive results from other enumerators
        if self.pipe:
            self.recv_thread = threading.Thread(target=self.receive_thread)
            self.recv_thread.start()

    def _build_deletion_order(self):
        """
        Constraint removal order heuristic (high impact first).

        If DIMACS group information is available, score each soft constraint by the
        total number of literals in its group clauses. Otherwise all scores are 0.
        """
        order = {i: 0 for i in range(1, self.n + 1)}
        groups = getattr(self.subs, "groups", None)
        dimacs = getattr(self.subs, "dimacs", None)
        if not groups or not dimacs:
            return order

        clause_lits = {}
        for ci, raw in enumerate(dimacs):
            try:
                tokens = raw.strip().split()
                lits = 0
                for tok in tokens:
                    val = int(tok)
                    if val != 0:
                        lits += 1
                clause_lits[ci] = lits
            except Exception:
                clause_lits[ci] = 0

        for idx in range(1, self.n + 1):
            group_clause_ids = groups.get(idx, [])
            score = 0
            for cid in group_clause_ids:
                score += clause_lits.get(cid, 0)
            order[idx] = -score
        return order

    def _record_known_mus(self, mus_set):
        key = frozenset(mus_set)
        if key in self.known_mus_keys:
            return
        self.known_mus_keys.add(key)
        self.known_muses.append(set(mus_set))

    def _update_core_intersection(self, seed_set, core_set):
        if not core_set:
            return
        core = set(core_set) & set(seed_set)
        if not core:
            return
        if self.core_isect is None:
            self.core_isect = set(core)
        else:
            self.core_isect &= core

    def _feedback_budget_exhausted(self):
        if not self.feedback_enabled:
            return True
        if self.feedback_max_clauses <= 0:
            return False
        return self.feedback_clause_count >= self.feedback_max_clauses

    def _learn_feedback(self, seed_is_sat, seed, known_max):
        # Learn from intermediate seeds only (if already known maximal/minimal, this is redundant).
        if known_max or self._feedback_budget_exhausted():
            return

        seedset = set(seed)
        key = frozenset(seedset)
        if seed_is_sat:
            clause_len = self.n - len(seedset)
            if self.feedback_sat_clause_max >= 0 and clause_len > self.feedback_sat_clause_max:
                return
            if key in self._feedback_sat_keys:
                return
            # Existing maximal SAT superset already implies this clause.
            if any(seedset <= old for old in self._feedback_sat_maximal):
                self.stats.increment_counter("feedback.sat.redundant")
                return

            # Drop dominated strict subsets.
            kept = []
            for old in self._feedback_sat_maximal:
                if old < seedset:
                    self._feedback_sat_keys.discard(frozenset(old))
                    self.stats.increment_counter("feedback.sat.prune")
                else:
                    kept.append(old)
            self._feedback_sat_maximal = kept

            self.map.block_down(seedset)
            self._feedback_sat_maximal.append(seedset)
            self._feedback_sat_keys.add(key)
            self.feedback_clause_count += 1
            self.stats.increment_counter("feedback.sat")
        else:
            clause_len = len(seedset)
            if self.feedback_unsat_clause_max >= 0 and clause_len > self.feedback_unsat_clause_max:
                return
            if key in self._feedback_unsat_keys:
                return
            # Existing minimal UNSAT subset already implies this clause.
            if any(old <= seedset for old in self._feedback_unsat_minimal):
                self.stats.increment_counter("feedback.unsat.redundant")
                return

            # Drop dominated strict supersets.
            kept = []
            for old in self._feedback_unsat_minimal:
                if seedset < old:
                    self._feedback_unsat_keys.discard(frozenset(old))
                    self.stats.increment_counter("feedback.unsat.prune")
                else:
                    kept.append(old)
            self._feedback_unsat_minimal = kept

            self.map.block_up(seedset)
            self._feedback_unsat_minimal.append(seedset)
            self._feedback_unsat_keys.add(key)
            self.feedback_clause_count += 1
            self.stats.increment_counter("feedback.unsat")

    def _probe_subset(self, subset, learn_feedback=False):
        """
        Probe a subset and request solver-side seed improvement (sat-subset / unsat-core).
        """
        probe = sorted(set(subset))
        with self.stats.time('smart.check'):
            is_sat, improved = self.subs.check_subset(probe, improve_seed=True)

        if improved is None:
            improved_set = set(probe)
        else:
            improved_set = set(int(x) for x in improved)
            if not improved_set and probe:
                improved_set = set(probe)

        if learn_feedback:
            self._learn_feedback(is_sat, improved_set if improved_set else probe, known_max=False)
        return is_sat, improved_set

    def _split_batch(self, batch):
        if len(batch) <= 1:
            return []
        ordered = sorted(batch, key=lambda i: self._deletion_order.get(i, 0))
        split = len(ordered) // 2
        left = set(ordered[:split])
        right = set(ordered[split:])
        out = []
        if left:
            out.append(left)
        if right:
            out.append(right)
        return out

    def _shrink_to_mus_deletion(self, seed_set, initial_core=None, locked=None):
        if initial_core:
            mus_set = set(initial_core) & set(seed_set)
            if not mus_set:
                mus_set = set(seed_set)
        else:
            mus_set = set(seed_set)

        locked_set = set(locked) if locked else set()
        ordered = sorted(mus_set, key=lambda i: self._deletion_order.get(i, 0))

        for idx in ordered:
            if idx not in mus_set or idx in locked_set:
                continue

            mus_set.remove(idx)
            if not mus_set:
                mus_set.add(idx)
                continue

            is_sat, improved = self._probe_subset(mus_set, learn_feedback=False)
            if is_sat:
                mus_set.add(idx)
            else:
                refined = set(improved) & mus_set if improved else set()
                if refined:
                    mus_set = refined
                    locked_set &= mus_set

        return mus_set

    def _preshrink_seed(self, seed_set, seed_core):
        projected = set(seed_set)
        for known_mus in sorted(self.known_muses, key=len):
            if known_mus <= seed_set:
                projected = set(known_mus)
                if seed_core:
                    projected |= (set(seed_core) & set(seed_set))
                break
        return projected

    def _shrink_to_mus_bicore_qx(self, seed_set, seed_core):
        projected_seed = self._preshrink_seed(seed_set, seed_core)

        if seed_core:
            current = set(seed_core) & projected_seed
            if not current:
                current = set(projected_seed)
        else:
            current = set(projected_seed)

        if not current:
            current = set(seed_set)

        locked = set(self.core_isect & current) if self.core_isect else set()
        batch_queue = []
        consecutive_sat = 0

        while True:
            unlocked = [i for i in current if i not in locked]
            if len(unlocked) <= self.core_handoff:
                break

            batch = None
            while batch_queue and batch is None:
                queued = batch_queue.pop()
                queued = {i for i in queued if i in current and i not in locked}
                if queued:
                    batch = queued

            if batch is None:
                available = sorted(unlocked, key=lambda i: self._deletion_order.get(i, 0))
                if not available:
                    break
                exp = min(consecutive_sat, self.core_backoff_cap)
                divisor = min(self.core_base_ratio * (2 ** exp), len(available))
                batch_size = max(1, len(available) // divisor)
                batch = set(available[:batch_size])

            test_set = set(current) - set(batch)
            if not test_set:
                if len(batch) == 1:
                    locked |= set(batch)
                else:
                    batch_queue.extend(self._split_batch(batch))
                continue

            is_sat, improved = self._probe_subset(test_set, learn_feedback=False)

            if not is_sat:
                consecutive_sat = 0
                refined = {i for i in improved if i in test_set} if improved else set()
                current = refined if refined else set(test_set)
                locked &= current
                continue

            consecutive_sat += 1
            if len(batch) == 1:
                locked |= set(batch)
            else:
                batch_queue.extend(self._split_batch(batch))

        residual = sorted(current, key=lambda i: self._deletion_order.get(i, 0))
        if not residual:
            return set()

        locked_residual = [i for i in residual if i in locked]
        unlocked_residual = [i for i in residual if i not in locked]

        def qx_unsat(enabled):
            is_sat, _ = self._probe_subset(enabled, learn_feedback=False)
            return not is_sat

        def qx_recursive(soft_part, hard_part, delta):
            if delta and qx_unsat(hard_part):
                return []
            if len(soft_part) == 1:
                return list(soft_part)

            split = len(soft_part) // 2
            left = soft_part[:split]
            right = soft_part[split:]

            delta2 = qx_recursive(right, hard_part + left, left)
            delta1 = qx_recursive(left, hard_part + delta2, delta2)
            return delta1 + delta2

        if unlocked_residual:
            qx_part = qx_recursive(unlocked_residual, list(locked_residual), [])
            mus_set = set(locked_residual) | set(qx_part)
        else:
            mus_set = set(locked_residual)

        return self._shrink_to_mus_deletion(mus_set)

    def _shrink_smart(self, seed_from_map, seed_core):
        seed_set = set(seed_from_map)
        if not seed_set:
            return set()

        core_set = set(seed_core) & seed_set if seed_core else None
        self._update_core_intersection(seed_set, core_set)
        mus_set = self._shrink_to_mus_bicore_qx(seed_set, core_set)

        if self.core_certify:
            with self.stats.time('smart.certify'):
                certified = self.subs.shrink(sorted(mus_set))
            if certified is None:
                self.stats.increment_counter("smart.certify.rejected")
                return None
            mus_set = set(certified)

        self._record_known_mus(mus_set)
        return mus_set

    def _set_bias(self, mus_bias):
        if self.bias_high == mus_bias:
            return
        self.bias_high = mus_bias
        self.config['bias'] = 'MUSes' if mus_bias else 'MCSes'
        self.stats.increment_counter("adaptive.switch.%s" % ("MUS" if mus_bias else "MCS"))

    def _maybe_activate_smart_core(self):
        if not self.portfolio or not self.smart_core_target or self.smart_core_active:
            return
        if self.mus_outputs < self.portfolio_smart_after_mus:
            return
        if self.total_outputs < self.portfolio_smart_after_outputs:
            return
        self.smart_core_active = True
        self.stats.increment_counter("portfolio.smart.enabled")

    def _record_result(self, result_type):
        self.total_outputs += 1
        if result_type == 'U':
            self.mus_outputs += 1
        self._maybe_activate_smart_core()

        if not self.adaptive:
            return
        self._recent_outputs.append(result_type)
        if len(self._recent_outputs) < self.adaptive_min_outputs:
            return

        mus_ratio = float(sum(1 for x in self._recent_outputs if x == 'U')) / len(self._recent_outputs)
        self.stats.add_stat("adaptive.window_mus_ratio", mus_ratio)

        low = max(0.0, self.adaptive_target_mus_ratio - self.adaptive_hysteresis)
        high = min(1.0, self.adaptive_target_mus_ratio + self.adaptive_hysteresis)
        if mus_ratio < low:
            # Too few MUS outputs recently: prioritize MUS-hunting.
            self._set_bias(True)
        elif mus_ratio > high:
            # Plenty of MUS outputs: diversify toward MCS-hunting.
            self._set_bias(False)

    def receive_thread(self):
        while self.pipe.poll(None):
            with self.stats.time('receive'):
                res = self.pipe.recv()
                if res == 'terminate':
                    # exit process on terminate message
                    os._exit(0)
                # Otherwise, we've received another result,
                # update blocking clauses.
                # Requires map solver to be thread-safe:
                assert hasattr(self.map, "__synchronized__") and self.map.__synchronized__

                if self.config['comms_ignore']:
                    continue

                if res[0] == 'S':
                    self.map.block_down(res[1])
                elif res[0] == 'U':
                    self.map.block_up(res[1])
                else:
                    assert False

    def record_delta(self, name, oldlen, newlen, up):
        if up:
            assert newlen >= oldlen
            self.stats.add_stat("delta.%s.up" % name, float(newlen - oldlen) / self.n)
        else:
            assert newlen <= oldlen
            self.stats.add_stat("delta.%s.down" % name, float(oldlen - newlen) / self.n)

    def enumerate(self):
        '''MUS/MCS enumeration with all the bells and whistles...'''

        for seed, known_max in self.seeds:
            seed = sorted(set(seed))
            seed_from_map = list(seed)

            if self.config['verbose']:
                print("- Initial seed: %s" % " ".join([str(x) for x in seed]))

            with self.stats.time('check'):
                # subset check may improve upon seed w/ unsat_core or sat_subset
                oldlen = len(seed)
                seed_is_sat, seed = self.subs.check_subset(seed, improve_seed=True)
                seed = sorted(set(seed))
                self.record_delta('checkA', oldlen, len(seed), seed_is_sat)
                known_max = (known_max and (seed_is_sat == self.bias_high))
                self._learn_feedback(seed_is_sat, seed, known_max)

            if self.config['verbose']:
                print("- Seed is %s." % {True: "SAT", False: "UNSAT"}[seed_is_sat])
                if known_max:
                    print("- Seed is known to be optimal.")
                else:
                    print("- Seed improved by check: %s" % " ".join([str(x) for x in seed]))

            if seed_is_sat:
                if known_max:
                    MSS = seed
                else:
                    with self.stats.time('grow'):
                        oldlen = len(seed)
                        MSS = self.subs.grow(seed)
                        self.record_delta('grow', oldlen, len(MSS), True)

                    if self.config['verbose']:
                        print("- Grow() -> MSS")

                with self.stats.time('block'):
                    res = ("S", MSS)
                    yield res

                    try:
                        self.subs.increment_MSS()
                    except AttributeError:
                        pass

                    self.map.block_down(MSS)
                    self._record_result(res[0])

                if self.config['verbose']:
                    print("- MSS blocked.")

            else:  # seed is not SAT
                self.got_top = True  # any unsat set covers the top of the lattice
                core_seed = set(seed)
                if known_max:
                    MUS = sorted(set(seed))
                    if self.smart_core_active:
                        self._update_core_intersection(seed_from_map, core_seed)
                        self._record_known_mus(set(MUS))
                else:
                    with self.stats.time('shrink'):
                        oldlen = len(seed)

                        if self.smart_core_active:
                            MUS = self._shrink_smart(seed_from_map, core_seed)
                        else:
                            MUS = self.subs.shrink(seed)

                        if MUS is None:
                            # seed was explored in another process
                            # in the meantime
                            self.stats.increment_counter("parallel_rejected")
                            continue
                        MUS = sorted(set(MUS))

                        self.record_delta('shrink', oldlen, len(MUS), False)

                    if self.config['verbose']:
                        print("- Shrink() -> MUS")

                with self.stats.time('block'):
                    res = ("U", MUS)
                    yield res

                    try:
                        self.subs.increment_MUS()
                    except AttributeError:
                        pass

                    self.map.block_up(MUS)
                    if self.smart_core_active and core_seed and set(MUS) != core_seed:
                        self.map.block_up(core_seed)
                        self.stats.increment_counter("smart.block.core")
                    self._record_result(res[0])

                if self.config['verbose']:
                    print("- MUS blocked.")

        if self.pipe:
            self.pipe.send(('complete', self.stats))
            self.recv_thread.join()


class SeedManager(object):
    def __init__(self, msolver, stats, config):
        self.map = msolver
        self.stats = stats
        self.config = config
        self._seed_queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        with self.stats.time('seed'):
            if not self._seed_queue.empty():
                return self._seed_queue.get()
            else:
                seed, known_max = self.seed_from_solver()
                if seed is None:
                    raise StopIteration
                return seed, known_max

    def add_seed(self, seed, known_max):
        self._seed_queue.put((seed, known_max))

    def seed_from_solver(self):
        known_max = self.config['maximize']
        return self.map.next_seed(), known_max
