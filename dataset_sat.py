import random
from pysat.solvers import Glucose3
import json
import argparse
import os

from pencil_utils import *
from tokenizer import *

##########################################
# Formatting SAT instances
##########################################

def format_instance(clauses, is_sat, reasoning_steps=None):
    """Format SAT instance with clauses, result, and optional reasoning steps."""
    text = format_cnf(clauses) + " <|endofprompt|> "
    text += " ".join(reasoning_steps) if reasoning_steps else str(is_sat)
    return text

def format_cnf(clauses):
    """Format CNF formula using ∨ for OR and ∧ for AND."""
    return " ∧ ".join(
        f"( {' ∨ '.join(str(lit) if lit > 0 else f'¬ {abs(lit)}' for lit in clause)} )"
        for clause in clauses
    )
    
##########################################
# The SATGenerator
# Generates random 3-SAT instances
##########################################

class SATGenerator:
    def __init__(self, min_vars=10, max_vars=20, clause_var_ratio=4.5, satisfiable_ratio=0.5):
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.clause_var_ratio = clause_var_ratio
        self.satisfiable_ratio = satisfiable_ratio
        self.MAX_ATTEMPTS = 5000

    def generate_dataset(self, num_samples, data_dir=None, format='io', save=True, batch_size=10000, buffer_size=1000):
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
                    is_sat, num_vars, instance_str = self.generate_instance(format=format)
                    item = {
                        'text': instance_str,
                        'label': is_sat,
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
                

    def generate_instance(self, num_variables=None, num_clauses=None, make_satisfiable=None, format='io', return_raw=False):
        """Generate a new 3-SAT instance."""
        # Set instance parameters
        num_variables = num_variables or random.randint(self.min_vars, self.max_vars)
        num_clauses = num_clauses or int(num_variables * self.clause_var_ratio)
        
        # Determine if instance should be satisfiable based on input or random choice
        if make_satisfiable is None:
            make_satisfiable = random.random() < self.satisfiable_ratio
            
        # Select variable IDs that will be used in the formula
        selected_var_ids = self._select_variable_ids(num_variables)
        
        # Generate instance based on satisfiability
        if make_satisfiable:
            # Generate a guaranteed satisfiable formula
            clauses, assignment = self._generate_satisfiable_instance(selected_var_ids, num_clauses)
        else:
            # Generate an unsatisfiable formula
            # Use trivial generator for clause_var_ratio=1 (for debugging), otherwise use standard UNSAT generator
            if self.clause_var_ratio == 1:
                clauses, assignment = self._generate_unsatisfiable_instance_trivial(selected_var_ids, num_clauses)
            else:
                clauses, assignment = self._generate_unsatisfiable_instance(selected_var_ids, num_clauses)

        # Format instance based on output format        
        if return_raw:
            return clauses
            
        if format in ['cot', 'pencil']:
            solver = SATSolver(num_variables, clauses)
            solver.solve(generate_cot=True)
            reasoning = solver.reasoning_steps
        else: 
            reasoning = None
            
        return make_satisfiable, num_variables, format_instance(clauses, make_satisfiable, reasoning)
    
    def _select_variable_ids(self, num_vars):
        """Randomly select variable IDs from range 1 to max_vars"""
        possible_vars = list(range(1, self.max_vars + 1))
        return random.sample(possible_vars, num_vars)

    def _generate_satisfiable_instance(self, selected_var_ids, num_clauses):
        """Generate a satisfiable 3-SAT instance."""
        # Create assignment dictionary for selected variables
        assignment = {var_id: random.randint(0, 1) for var_id in selected_var_ids}
        clauses = []
        
        while len(clauses) < num_clauses:
            # Select 3 random variables from our chosen set
            vars_in_clause = random.sample(selected_var_ids, 3)
            negations = [random.choice([-1, 1]) for _ in range(3)]
            
            # Ensure at least one literal satisfies the clause
            satisfied = False
            for i, var in enumerate(vars_in_clause):
                if (assignment[var] == 1 and negations[i] == 1) or (assignment[var] == 0 and negations[i] == -1):
                    satisfied = True
                    break
            
            if not satisfied:
                idx = random.randint(0, 2)
                negations[idx] = -negations[idx]
            
            clause = [var * neg for var, neg in zip(vars_in_clause, negations)]
            clauses.append(clause)
        
        return clauses, assignment

    def _generate_unsatisfiable_instance(self, selected_var_ids, num_clauses):
        """Generate an unsatisfiable 3-SAT instance."""
        for attempt in range(self.MAX_ATTEMPTS):
            clauses = []
            
            while len(clauses) < num_clauses:
                vars_in_clause = random.sample(selected_var_ids, 3)
                negations = [random.choice([-1, 1]) for _ in range(3)]
                clause = [v * n for v, n in zip(vars_in_clause, negations)]
                clauses.append(clause)
            
            # Verify unsatisfiability
            probability_upper_bound = (2 * (7/8)**(num_clauses/len(selected_var_ids)))**len(selected_var_ids)
            if probability_upper_bound < 0.001 or self._verify_unsat(clauses):
                return clauses, None
            
        raise RuntimeError(f"Failed to generate verified unsatisfiable instance after {self.MAX_ATTEMPTS} attempts")
        
    def _generate_unsatisfiable_instance_trivial(self, selected_var_ids, num_clauses):
        """Generate unsatisfiable 3-SAT with shuffled variable IDs."""
        clauses = []
        
        # Use first two variables from our selected set for the unsatisfiable core
        x1, x2 = selected_var_ids[:2]
        sign = random.choice([-1, 1])
        
        # Create minimal unsatisfiable core
        clauses.append([sign * x1, sign * x1, sign * x1])
        clauses.append([-sign * x1, sign * x2, sign * x2])
        clauses.append([-sign * x2, -sign * x1, -sign * x1])
        
        # Add random clauses until reaching num_clauses
        while len(clauses) < num_clauses:
            vars_in_clause = random.sample(selected_var_ids, 3)
            negations = [random.choice([-1, 1]) for _ in range(3)]
            clause = [v * n for v, n in zip(vars_in_clause, negations)]
            clauses.append(clause)
        
        return clauses, None

    def _verify_unsat(self, clauses):
        """Verify if a formula is unsatisfiable using a SAT solver."""
        solver = Glucose3()
        for clause in clauses:
            solver.add_clause(clause)
        is_sat = solver.solve()
        solver.delete()
        return not is_sat

##########################################
# SAT Solver for Generating Chain-of-Thought (with special tokens for training PENCIL)
##########################################
class SATSolver:
    def __init__(self, num_vars=None, clauses=None):
        """Initialize SAT solver with number of variables and list of clauses."""
        self.num_vars = num_vars
        self.clauses = clauses # This is the initial formula that stays static
        self.reasoning_steps = []
        
    def solve(self, generate_cot=False):
        # Initialize variables for generate CoT
        self.reasoning_steps = []
        
        # Run the DPLL algorithm with local assignment
        is_sat = self([clause[:] for clause in self.clauses])
        
        if generate_cot:
            return is_sat, format_instance(self.clauses, is_sat, self.reasoning_steps)
        return is_sat
        
    def __call__(self, clauses):
        self._add_step(PENCIL_TOKENS['call']) # Function begins
        self._log_state(clauses) # Description of the Input
        
        is_sat = self._dpll(clauses)
        
        self._add_step(PENCIL_TOKENS['sep']) # Separator
        self._log_state(clauses, is_sat)
        self._add_step(PENCIL_TOKENS['return']) # Function returns
        
        return is_sat

    def _dpll(self, clauses):
        # Case 1 - Found unit clause with literal, setting variable
        var, value = self._find_unit_clause(clauses)
        if var is not None: 
            self._add_step(f"Found {'¬ ' if not value else ''}{var + 1} Let {var + 1} = {str(value)}")
            
            new_clauses = self._simplify_clauses(clauses, var + 1, value)
            is_sat = self._check_base_case(new_clauses)
            if is_sat is None: is_sat = self(new_clauses)
            
            return is_sat

        # Case 2 - If no unit clauses, branch on a variable
        var = self._find_unassigned_variable(clauses)
        for value in [True, False]:
            self._add_step(f"Try {var + 1} = {str(value)}")
            
            new_clauses = self._simplify_clauses(clauses, var + 1, value)
            is_sat = self._check_base_case(new_clauses)
            if is_sat is None: is_sat = self(new_clauses)
                
            if is_sat: return True
        
        return False

    def _simplify_clauses(self, clauses, var, value):
        """Simplify clauses based on variable assignment."""
        result = []
        for clause in clauses:
            # If the literal matches the assignment, skip the clause (it's satisfied)
            if (var in clause and value) or (-var in clause and not value):
                continue
            # Remove the falsified literal and keep others
            new_clause = [lit for lit in clause if lit != var and lit != -var]            
            result.append(new_clause)
    
        return result
    
    def _find_unit_clause(self, clauses):
        """Find and return a unit clause if one exists."""
        for clause in clauses:
            if len(clause) == 1:
                return abs(clause[0]) - 1, clause[0] > 0  # Return variable and value
        return None, None

    def _find_unassigned_variable(self, clauses):
        """Find the smallest unassigned variable from the clauses."""
        unassigned_variables = set()
        
        for clause in clauses:
            for literal in clause:
                variable = abs(literal)
                unassigned_variables.add(variable)
        
        if unassigned_variables:
            return min(unassigned_variables) - 1
        return None
    
    def _check_base_case(self, clauses):
        """Handle base cases of DPLL algorithm."""
        if len(clauses) == 0: return True
        if any(not c for c in clauses): return False
        return None

    def _add_step(self, detail):
        self.reasoning_steps.append(detail)
        
    def _log_state(self, clauses, is_sat = None):
        if is_sat is None:
            self._add_step(f"Question: {format_cnf(clauses)}")
        else:
            self._add_step(f"Answer: {str(is_sat)}")
        
##########################################
# External SAT Solver for Verification
##########################################
def test_sat_solver(Solver, num_tests=10000, num_variables=10, num_clauses=40):
    """Test SAT solver with both satisfiable and unsatisfiable instances."""
    generator = SATGenerator()
    false_neg = 0  # SAT claimed as UNSAT
    false_pos = 0  # UNSAT claimed as SAT

    for i in range(num_tests):
        is_satisfiable = (i % 2 == 0)
        clauses = generator.generate_instance(
            num_variables=num_variables, 
            num_clauses=num_clauses,
            make_satisfiable=is_satisfiable,
            return_raw=True
        )
        
        solver = Solver(num_variables, clauses)
        solver_sat = solver.solve()
        
        if is_satisfiable and not solver_sat:
            false_neg += 1
        elif not is_satisfiable and solver_sat:
            false_pos += 1

    total_errors = false_neg + false_pos
    print(f"Tests: {num_tests}, False negatives: {false_neg}, False positives: {false_pos}")
    print(f"Success rate: {100 * (num_tests - total_errors) / num_tests:.2f}%")
    
    return total_errors == 0  # Return true if no errors occurred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-SAT Problem Training Dataset Generator")
    parser.add_argument('--num_samples', type=int, default=1000000, help='Number of samples to generate')
    parser.add_argument('--train_size', type=int, default=100000, help='Size of training set')
    parser.add_argument('--val_size', type=int, default=1000, help='Size of validation set')
    parser.add_argument('--test_size', type=int, default=1000, help='Size of test set')
    parser.add_argument('--data_dir', type=str, default='data/sat', help='Output JSONL file path')
    parser.add_argument('--satisfiable_ratio', type=float, default=0.5, help='Ratio of satisfiable instances')
    parser.add_argument('--min_vars', type=int, default=10, help='Minimum number of variables')
    parser.add_argument('--max_vars', type=int, default=20, help='Maximum number of variables')
    parser.add_argument('--clause_var_ratio', type=float, default=4.3, help='Ratio of clauses to variables')
    parser.add_argument('--format', type=str, default='pencil', choices=['io', 'cot', 'pencil'], help='Output format')
    args = parser.parse_args()

    # assert test_sat_solver(SATSolver, num_tests=1000, num_variables=10, num_clauses=45) # uncomment to test the correctness of the data

    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Step 1: Generate dataset
    sat = SATGenerator(
        min_vars=args.min_vars, 
        max_vars=args.max_vars, 
        clause_var_ratio=args.clause_var_ratio,
        satisfiable_ratio=args.satisfiable_ratio
    )
    
    data = sat.generate_dataset(
        num_samples=args.num_samples, 
        data_dir=args.data_dir, 
        format=args.format,
        save=True # False to skip saving and only return data
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
    if args.format == 'pencil':
        process_pencil_data(args.data_dir, tokenizer)