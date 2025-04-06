import random
import json
import argparse
import os
from copy import deepcopy

from pencil_utils import *
from tokenizer import *

##########################################
# Formatting QBF instances
##########################################

def format_qbf(prefix, clauses):
    """Simple textual representation, e.g. ∃1 ∀2 : #1 (1 ∨ ¬2) #2 (¬1 ∨ 2)..."""
    prefix_str = " ".join(f"{q} {v}" for (q, v) in prefix)

    def lit_str(l):
        return f"¬ {abs(l)}" if l < 0 else str(l)

    def clause_str(c, i):
        return f"#{i+1} ( " + " ∨ ".join(lit_str(x) for x in c) + " )"

    cnf_str = " ".join(clause_str(c, i) for i,c in enumerate(clauses))
    return f"{prefix_str} : {cnf_str}"

def format_clause(clause):
    def lit_str(l):
        return f"¬ {abs(l)}" if l<0 else str(l)
    return "( " + " ∨ ".join(lit_str(x) for x in clause) + " )"

def format_assignment(assignment):
    if not assignment:
        return "{ }"
    parts = [f"{var} = {value}" for var, value in sorted(assignment.items())]
    return " ".join(parts)

def format_instance(prefix, clauses, is_valid, reasoning_steps=None):
    text = format_qbf(prefix, clauses) + " <|endofprompt|> "
    text += " ".join(reasoning_steps) if reasoning_steps else str(is_valid)
    return text

##########################################
# QBF Generator 
###########################################

class QBFGenerator:
    def __init__(self, min_vars=3, max_vars=6, clause_var_ratio=2.0,
                 valid_ratio=0.5, existential_prob=0.5):
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.clause_var_ratio = clause_var_ratio
        self.valid_ratio = valid_ratio
        self.existential_prob = existential_prob
        self.MAX_ATTEMPTS = 5000

    def generate_dataset(self, num_samples, data_dir=None, format='io',
                         save=True, batch_size=10000, buffer_size=1000):
        if save and not data_dir:
            raise ValueError("data_dir must be specified when save=True")

        data = [] if not save else None
        file_handle = None
        buffer = []
        
        try:
            if save:
                os.makedirs(data_dir, exist_ok=True)
                output_file = os.path.join(data_dir, 'data.jsonl')
                file_handle = open(output_file, 'w', buffering=8192, encoding='utf-8')
            
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                
                for i in range(batch_start, batch_end):
                    is_valid, num_vars, instance_str = self.generate_instance(format=format)
                    item = {
                        'text': instance_str,
                        'label': is_valid,
                        'size': num_vars
                    }
                    
                    if save:
                        buffer.append(json.dumps(item, ensure_ascii=False))
                        if len(buffer) >= buffer_size:
                            file_handle.write('\n'.join(buffer) + '\n')
                            buffer.clear()
                    else:
                        data.append(item)
                        
                    if (i + 1) % 1000 == 0:
                        print(f"Generated {i + 1}/{num_samples} samples...")
                
                # Clear memory after each batch
                if save and buffer:
                    file_handle.write('\n'.join(buffer) + '\n')
                    buffer.clear()
                    
        finally:
            if file_handle:
                if buffer:  # Write remaining items
                    file_handle.write('\n'.join(buffer) + '\n')
                file_handle.close()
                
        return data

    def generate_instance(self, num_variables=None, num_clauses=None,
                          make_valid=None, format='io', return_raw=False):
        """Produce either a valid or invalid QBF, then add uniform advanced filler, rename."""
        # 1) Decide #vars, #clauses
        n = num_variables or random.randint(self.min_vars, self.max_vars)
        num_clauses = num_clauses or int(n * self.clause_var_ratio)

        # 2) Decide valid or invalid
        if make_valid is None:
            make_valid = (random.random() < self.valid_ratio)

        # 3) Pick a random filler_count. We'll add the *same* count to both valid & invalid
        filler_count = random.randint(0,3)

        # 4) Build aggregator-based valid or invalid
        if make_valid:
            prefix, clauses = self._build_valid_instance(n, num_clauses)
        else:
            prefix, clauses = self._build_invalid_instance(n, num_clauses)

        # 5) Insert advanced filler
        clauses = self._add_advanced_filler(prefix, clauses, filler_count, n)

        # 6) Rename / reorder variables
        prefix, clauses = self._rename_variables(prefix, clauses)

        if return_raw:
            return prefix, clauses

        # Possibly chain-of-thought
        if format in ['cot','pencil']:
            solver = QBFSolver(prefix, clauses)
            solver.solve(generate_cot=True)
            reasoning = solver.reasoning_steps
        else:
            reasoning = None

        return make_valid, n, format_instance(prefix, clauses, make_valid, reasoning)

    #   Build a "Truly Valid" instance
    def _build_valid_instance(self, n, num_clauses):
        """Try aggregator expansions + solver check to ensure validity."""
        for _ in range(self.MAX_ATTEMPTS):
            prefix, clauses = self._try_build_aggregator_valid(n, num_clauses)
            solver = NaiveQBFSolver(prefix, clauses)
            if solver.is_valid():
                return prefix, clauses
        raise RuntimeError("Failed to build a valid QBF after max attempts.")

    def _try_build_aggregator_valid(self, n, num_clauses):
        var_ids = list(range(1, n+1))
        random.shuffle(var_ids)
        prefix = []
        for v in var_ids:
            if random.random()<self.existential_prob:
                prefix.append(('∃', v))
            else:
                prefix.append(('∀', v))

        # ensure at least 1∃ and 1∀ if n>1
        if n>1:
            if not any(q=='∃' for q,_ in prefix):
                prefix[0] = ('∃', prefix[0][1])
            if not any(q=='∀' for q,_ in prefix):
                prefix[-1] = ('∀', prefix[-1][1])

        # expansions
        clauses = []
        earlier_univs = []
        for (q,v) in prefix:
            if q=='∀':
                earlier_univs.append(v)
            else:
                sub = self._encode_exvar_aggregator(v, earlier_univs)
                clauses.extend(sub)

        # fill up with random tautologies if needed
        while len(clauses)<num_clauses:
            c = self._make_random_tautology(n)
            clauses.append(c)
        if len(clauses)>num_clauses:
            clauses = clauses[:num_clauses]

        return prefix, clauses

    def _encode_exvar_aggregator(self, e, univs):
        """ e <-> aggregator(all univs, random sign), aggregator in {AND, OR}. """
        if not univs:
            return [[e]]

        aggregator = random.choice(['AND','OR'])
        litlist = []
        for u in univs:
            sign = random.choice([True,False])
            litlist.append(u if sign else -u)

        if aggregator=='AND':
            # e => each lit => (¬e ∨ lit)
            # all-lits => e => (¬lit1 ∨ ¬lit2 ∨ ... ∨ e)
            cs = []
            for L in litlist:
                cs.append([-e, L])
            big = []
            for L in litlist:
                big.append( -abs(L) if L>0 else abs(L) )
            big.append(e)
            cs.append(big)
            return cs
        else:
            # aggregator=='OR'
            # e => (lit1 ∨ lit2 ∨ ...)
            # each lit => e
            cs = []
            big_disj = [-e]+litlist
            cs.append(big_disj)
            for L in litlist:
                c2 = [ -abs(L) if L>0 else abs(L), e ]
                cs.append(c2)
            return cs

    def _make_random_tautology(self, n):
        """2..3-literal guaranteed true: x ∨ ¬x ∨ ±y?"""
        length = random.randint(2,3)
        x = random.randint(1,n)
        x_sign = random.choice([True,False])
        c = [(x if x_sign else -x), (-x if x_sign else x)]
        while len(c)<length:
            y = random.randint(1,n)
            s = random.choice([True,False])
            c.append(y if s else -y)
        return c

    #   Build an "Invalid" instance (perturb a valid)
    def _build_invalid_instance(self, n, num_clauses):
        prefix, clauses = self._build_valid_instance(n, num_clauses)
        for _ in range(self.MAX_ATTEMPTS):
            new_clauses = self._random_perturb(clauses)
            solver = NaiveQBFSolver(prefix, new_clauses)
            if not solver.is_valid():
                return prefix, new_clauses
            clauses = new_clauses
        raise RuntimeError("Could not break aggregator-based valid after many tries")

    def _random_perturb(self, clauses):
        new = deepcopy(clauses)
        methods = ["flip_lit","remove_lit","remove_clause","add_clause"]
        m = random.choice(methods)

        if m=="flip_lit" and new:
            cidx = random.randrange(len(new))
            if new[cidx]:
                lidx = random.randrange(len(new[cidx]))
                oldl = new[cidx][lidx]
                new[cidx][lidx] = -oldl
        elif m=="remove_lit" and new:
            cidx = random.randrange(len(new))
            if len(new[cidx])>1:
                lidx = random.randrange(len(new[cidx]))
                del new[cidx][lidx]
            else:
                new = self._flip_fallback(new)
        elif m=="remove_clause" and new:
            cidx = random.randrange(len(new))
            del new[cidx]
        else:
            # add_clause
            c = []
            length = random.randint(1,3)
            all_vars = set()
            for cl in new:
                for lit in cl:
                    all_vars.add(abs(lit))
            if not all_vars:
                all_vars = {1}
            chosen = random.sample(list(all_vars), k=min(len(all_vars), length))
            for v in chosen:
                s = random.choice([True,False])
                c.append(v if s else -v)
            new.append(c)
        return new

    def _flip_fallback(self, clauses):
        new = deepcopy(clauses)
        if not new:
            return new
        cidx = random.randrange(len(new))
        if not new[cidx]:
            return new
        lidx = random.randrange(len(new[cidx]))
        new[cidx][lidx] = -new[cidx][lidx]
        return new

    #   3) Add advanced filler
    def _add_advanced_filler(self, prefix, clauses, filler_count, n):
        """
        Insert exactly 'filler_count' advanced filler clauses.
        We do a 50% chance to create a random tautology, 
        otherwise a random normal (non-tautology) clause.
        """
        new_clauses = deepcopy(clauses)
        for _ in range(filler_count):
            if random.random()<0.5:
                # advanced random tautology
                new_clauses.append(self._make_tautology_or_multi(n))
            else:
                # random normal clause
                new_clauses.append(self._make_random_clause(n))
        return new_clauses

    def _make_tautology_or_multi(self, n):
        """
        Possibly a bigger tautology: pick 2..4 distinct variables,
        ensure at least one pair x,¬x is present.
        """
        length = random.randint(2,4)
        chosen = random.sample(range(1,n+1), k=min(n,length))
        c = []
        # forcibly ensure one var x plus ¬x
        x = random.choice(chosen)
        # random sign for x
        x_sign = random.choice([True,False])
        c.append(x if x_sign else -x)
        c.append(-x if x_sign else x)
        # if we still have space, add more random
        while len(c)<length:
            y = random.randint(1,n)
            s = random.choice([True,False])
            c.append(y if s else -y)
        random.shuffle(c)
        return c

    def _make_random_clause(self, n):
        """A random (non-tautological) clause with 2..4 distinct variables, random signs."""
        length = random.randint(2,4)
        available = list(range(1,n+1))
        random.shuffle(available)
        chosen = available[:length]
        c = []
        for v in chosen:
            s = random.choice([True,False])
            c.append(v if s else -v)
        return c

    #   4) Rename / reorder variables
    def _rename_variables(self, prefix, clauses):
        """
        1) Gather var IDs from prefix
        2) Make a random permutation
        3) Re-label prefix & clauses
        4) Shuffle final clauses
        """
        new_prefix = deepcopy(prefix)
        new_clauses = deepcopy(clauses)

        used = set(v for (q,v) in new_prefix)
        used_vars = list(used)
        random.shuffle(used_vars)

        var_map = {}
        i = 1
        for ov in used_vars:
            var_map[ov] = i
            i+=1

        # rename prefix
        for idx in range(len(new_prefix)):
            q, oldv = new_prefix[idx]
            newv = var_map[oldv]
            new_prefix[idx] = (q, newv)

        # rename clauses
        for cidx in range(len(new_clauses)):
            for lidx in range(len(new_clauses[cidx])):
                oldlit = new_clauses[cidx][lidx]
                sign = (oldlit>0)
                ov = abs(oldlit)
                nv = var_map.get(ov,1)
                new_clauses[cidx][lidx] = (nv if sign else -nv)

        # reorder final clauses
        random.shuffle(new_clauses)
        return new_prefix, new_clauses


##########################################
# QBF Solver for Generating Chain-of-Thought
##########################################
class QBFSolver:
    def __init__(self, prefix, clauses):
        self.prefix = prefix
        self.clauses = clauses
        self.reasoning_steps = []

    def solve(self, generate_cot=False):
        self.reasoning_steps = []
        is_valid = self(prefix_index=0, assignment={})
        if generate_cot:
            return is_valid, format_instance(self.prefix, self.clauses, is_valid, self.reasoning_steps)
        return is_valid

    def __call__(self, prefix_index, assignment):
        self._add_step(PENCIL_TOKENS['call'])
        self._log_state(prefix_index, assignment)
        is_valid = self._qbf(prefix_index, assignment)
        self._add_step(PENCIL_TOKENS['sep'])
        self._log_state(prefix_index, assignment, is_valid)
        self._add_step(PENCIL_TOKENS['return'])
        return is_valid

    def _qbf(self, prefix_index, assignment):
        if prefix_index == len(self.prefix):
            return self._evaluate_clauses(assignment)

        quant, var_id = self.prefix[prefix_index]
        if quant=='∃':
            for choice in [False,True]:
                self._add_step(f"Try {var_id} = {choice}")
                new_assign = assignment.copy()
                new_assign[var_id] = choice
                if self(prefix_index+1, new_assign):
                    return True
            return False
        else:
            # '∀'
            for choice in [False,True]:
                self._add_step(f"Try {var_id} = {choice}")
                new_assign = assignment.copy()
                new_assign[var_id] = choice
                if not self(prefix_index+1, new_assign):
                    return False
            return True

    def _evaluate_clauses(self, assignment):
        for i,c in enumerate(self.clauses):
            clause_ok = False
            self._add_step(f"Check #{i} {format_clause(c)}")
            for lit in c:
                var_id = abs(lit)
                sign   = (lit>0)
                if assignment.get(var_id, False)==sign:
                    self._add_step("True")
                    clause_ok = True
                    break
            if not clause_ok:
                self._add_step("False")
                return False
        self._add_step("Formula = True")
        return True

    def _add_step(self, msg):
        self.reasoning_steps.append(msg)

    def _log_state(self, prefix_index, assignment, is_valid=None):
        if is_valid is None:
            if prefix_index<len(self.prefix):
                q,v = self.prefix[prefix_index]
                self._add_step(f"Question: prefix_from {q} {v}")
            else:
                self._add_step(f"Question: evaluate {format_assignment(assignment)}")
        else:
            self._add_step(f"Answer: {is_valid}")


##########################################
# A Naive QBF Solver for Verification
##########################################
class NaiveQBFSolver:
    """
    Checks if a QBF formula is valid (True) or invalid (False).
    prefix: list of (quant, varID)
    clauses: list of lists of ints
    """
    def __init__(self, prefix, clauses):
        self.prefix = prefix
        self.clauses = clauses

    def is_valid(self):
        return self._check_qbf(0, {})

    def _check_qbf(self, idx, assignment):
        if idx==len(self.prefix):
            return self._evaluate_clauses(assignment)

        quant, var_id = self.prefix[idx]
        if quant=='∃':
            for choice in [False,True]:
                assignment[var_id] = choice
                if self._check_qbf(idx+1, assignment):
                    return True
            return False
        else:
            # '∀'
            for choice in [False,True]:
                assignment[var_id] = choice
                if not self._check_qbf(idx+1, assignment):
                    return False
            return True

    def _evaluate_clauses(self, assignment):
        for c in self.clauses:
            satisfied = False
            for lit in c:
                var_id = abs(lit)
                sign   = (lit>0)
                if assignment.get(var_id, False)==sign:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="QBF Problem Dataset Generator")
    parser.add_argument('--num_samples', type=int, default=1000000, help='Number of samples to generate')
    parser.add_argument('--train_size', type=int, default=100000, help='Size of training set')
    parser.add_argument('--val_size', type=int, default=1000, help='Size of validation set')
    parser.add_argument('--test_size', type=int, default=1000, help='Size of test set')
    parser.add_argument('--data_dir', type=str, default='data/qbf', help='Output directory for JSONL')
    parser.add_argument('--valid_ratio', type=float, default=0.5, help='Ratio of valid QBF formulas')
    parser.add_argument('--min_vars', type=int, default=3, help='Minimum number of variables')
    parser.add_argument('--max_vars', type=int, default=6, help='Maximum number of variables')
    parser.add_argument('--clause_var_ratio', type=float, default=2.0, help='Approx ratio #clauses to #vars')
    parser.add_argument('--format', type=str, default='pencil', choices=['io','cot','pencil'], help='Output format')
    parser.add_argument('--existential_prob', type=float, default=0.5, help='Prob a variable is existential')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Step 1: Generate dataset
    qbf_gen = QBFGenerator(
        min_vars=args.min_vars,
        max_vars=args.max_vars,
        clause_var_ratio=args.clause_var_ratio,
        valid_ratio=args.valid_ratio,
        existential_prob=args.existential_prob
    )

    data = qbf_gen.generate_dataset(
        num_samples=args.num_samples,
        data_dir=args.data_dir,
        format=args.format,
        save=True
    )

    # Step 2: Build tokenizer from the dataset
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab_from_file(args.data_dir)
    tokenizer.save_meta_to_file(args.data_dir)

    # Step 3: Split dataset
    process_dataset(args.data_dir,
                    train_size=args.train_size, 
                    val_size=args.val_size,
                    test_size=args.test_size,
                    tokenizer=tokenizer)
    
    # (Optional) Step 4: Generate pencil training data
    if args.format=='pencil':
        process_pencil_data(args.data_dir, tokenizer)
        