from dataclasses import dataclass
from typing import Dict, List, Set, Callable, Tuple
import json
import time
import copy
import argparse
import os

from pencil_utils import *
from tokenizer import *

CATEGORY_NAMES_5 = ["Color", "Nationality", "Beverage", "Cigarette brand", "Pet"]
CATEGORY_NAMES_4 = ["Color", "Nationality", "Beverage", "Pet"]
CATEGORY_NAMES_3 = ["Color", "Nationality", "Pet"]

PUZZLES = {
    3: [
        ["Red", "Green", "Blue"],
        ["Brit", "Swede", "German"],
        ["Dogs", "Birds", "Fish"]
    ],
    4: [
        ["Red", "Green", "Blue", "Yellow"],
        ["Brit", "Swede", "Dane", "German"],
        ["Tea", "Coffee", "Milk", "Beer"],
        ["Dogs", "Birds", "Cats", "Fish"]
    ],
    5: [
        ["Red", "Green", "White", "Yellow", "Blue"],
        ["Brit", "Swede", "Dane", "Norwegian", "German"],
        ["Tea", "Coffee", "Milk", "Beer", "Water"],
        ["Pall Mall", "Dunhill", "Blend", "Blue Master", "Prince"],
        ["Dogs", "Birds", "Cats", "Horses", "Fish"]
    ]
}

def make_constraint_function(relation: str, col_count: int) -> Tuple[Callable, str]:
    """Create constraint comparison function & label"""
    if relation == "same":
        return (lambda *args: len(args) == 2 and args[0] == args[1], "SAME")
    elif relation == "left":
        return (lambda *args: len(args) == 2 and args[0] == args[1] - 1, "LEFT")
    elif relation == "right":
        return (lambda *args: len(args) == 2 and args[0] == args[1] + 1, "RIGHT")
    elif relation == "far_left":
        return (lambda *args: len(args) == 1 and args[0] == 0, "FAR_LEFT")
    elif relation == "far_right":
        return (lambda *args: len(args) == 1 and args[0] == col_count - 1, "FAR_RIGHT")
    elif relation == "middle":
        middle = col_count // 2
        return (lambda *args: len(args) == 1 and args[0] == middle, "MIDDLE")
    else:
        return (lambda *args: False, f"UNKNOWN_{relation}")

@dataclass
class Constraint:
    attr_indices: List[int]
    values: List[str]
    compare: Callable
    debug_text: str
    relation_name: str  # e.g. "LEFT", "RIGHT", "SAME"

def format_table(table: List[List[str]]) -> str:
    """Format solution table with aligned columns for the final output"""
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    lines = []
    for line in table:
        lines.append("| " + " | ".join(f"{x:{col_width[i]}}" for i, x in enumerate(line)) + " |")
    return " \n ".join(lines)

def format_ranges_nl(ranges: List[List[Set[str]]], cat_names: List[str]) -> str:
    """Return a user-friendly textual representation of puzzle state"""
    lines = []
    num_houses = len(ranges[0]) if ranges else 0
    for house_idx in range(num_houses):
        lines.append(f"House#{house_idx + 1}")
        for cat_idx, cat_label in enumerate(cat_names):
            items = sorted(ranges[cat_idx][house_idx])
            if items:
                if len(items) == 1:
                    lines.append(f"{cat_label} category is {items[0]}")
                else:
                    lines.append(f"{cat_label} category have {len(items)} possibilities {' '.join(items)}")
            else:
                lines.append(f"{cat_label} category is empty")
    return " \n ".join(lines)

def _print_diff(old_ranges: List[List[Set[str]]],
                new_ranges: List[List[Set[str]]],
                cat_names: List[str]) -> List[str]:
    """Compare old puzzle vs new puzzle, returning lines describing changed houses/categories only"""
    diff_lines = []
    if not old_ranges or not new_ranges:
        return diff_lines

    num_cats = len(old_ranges)
    for cat_idx in range(num_cats):
        cat_label = cat_names[cat_idx]
        old_row = old_ranges[cat_idx]
        new_row = new_ranges[cat_idx]
        for house_idx, (old_cell, new_cell) in enumerate(zip(old_row, new_row)):
            if old_cell != new_cell:
                old_sorted = sorted(old_cell)
                new_sorted = sorted(new_cell)
                old_str = " ".join(old_sorted) if old_sorted else "empty"
                new_str = " ".join(new_sorted) if new_sorted else "empty"
                diff_lines.append(
                    f"House#{house_idx + 1} {cat_label} category changed from {len(old_sorted)} possibilities {old_str} to {len(new_sorted)} possibilities {new_str} "
                    # f"House#{house_idx + 1} {cat_label} category changed to {len(new_sorted)} possibilities {new_str} "
                )
    return diff_lines

def _summarize_choices_count(ranges: List[List[Set[str]]], cat_names: List[str]) -> str:
    """
    Summarize how many possibilities remain in each house/category (just the count).
    Example output:
      House #1 => Color:2, Nationality:1, Pet:2
    """
    lines = []
    if not ranges or not ranges[0]:
        return "Puzzle is invalid"

    num_houses = len(ranges[0])
    for house_idx in range(num_houses):
        lines.append(f"House#{house_idx+1}")
        for cat_idx, cat_label in enumerate(cat_names):
            count = len(ranges[cat_idx][house_idx])
            lines.append(f"{cat_label} category have {count} possibilities")
    return " \n ".join(lines)


def is_constraint_satisfied(ranges: List[List[Set[str]]],
                            c: Constraint) -> bool:
    """
    Check if constraint 'c' is already satisfied by the puzzle's ranges.
    For example, if c says 'Dogs LEFT German', 
    we see if there's exactly one house that is Dogs and exactly one house that's German,
    and confirm they satisfy that relation.

    If we can't confirm or deny because the puzzle is not fully determined, return False (not satisfied).
    If we can confirm that the puzzle definitely satisfies the relation, return True.
    """

    item_positions = []
    for i_item, row_i in enumerate(c.attr_indices):
        item = c.values[i_item]
        row_ranges = ranges[row_i]

        # find all houses where item could be
        possible_cols = [col_i for col_i, sset in enumerate(row_ranges) if item in sset]
        if len(possible_cols) == 1:
            # pinned exactly to col
            item_positions.append(possible_cols[0])
        else:
            # if 0 or >1 => can't confirm
            return False

    # now we have exactly one col for c.values[0], exactly one col for c.values[1], ...
    # apply c.compare(*item_positions)
    return c.compare(*item_positions)


@dataclass
class EinsteinSolver:
    table: List[List[str]]

    def __post_init__(self):
        self.num_houses = len(self.table[0])
        self.num_categories = len(self.table)
        self.reasoning_steps: List[str] = []
        self.depth = 0

    def _add_step(self, msg: str):
        """Store a log message"""
        self.reasoning_steps.append(msg)

    def solve(self, constraints: Dict[str, Constraint]) -> Tuple[bool, List[List[List[Set[str]]]]]:
        """
        Build puzzle state, then call _solve_puzzle. 
        We now store constraints in a dictionary => if a constraint is satisfied, we remove it from the dict.
        """
        
        # Add the prompt
        total_constraints = len(constraints)
        for idx, (cid, c) in enumerate(constraints.items()):
            # Add endofprompt marker if this is the last constraint
            end_marker = " <|endofprompt|>" if idx == total_constraints - 1 else ""
            self._add_step(f"Constraint#{int(cid)+1} : {c.debug_text}{end_marker}")
        
        # Build puzzle state
        puzzle_ranges = [[set(row) for _ in range(self.num_houses)] for row in self.table]

        # Recur
        solutions = self._solve_puzzle(puzzle_ranges, constraints)
        
        # If at least one solution => figure out who owns the fish
        fish_owner_nationality = None
        fish_owner_house = None
        if solutions:
            first_solution = solutions[0]
            # The "Pets" row is row=4 if we have 5 categories, else row=2 if we have 3 categories
            pet_row_idx = {
                3: 2,  # 3×3 puzzle
                4: 3,  # 4×4 puzzle
                5: 4   # 5×5 puzzle
            }[self.num_categories]
            # The "Nationality" row is row=1 (both for 3-cat and 5-cat puzzle structures)
            nationality_row_idx = 1
            
            # Find which house (column) actually has "Fish"
            for col in range(self.num_houses):
                if "Fish" in first_solution[pet_row_idx][col]:
                    # This house definitely has "Fish" => check nationality in same column
                    # by design, there should be exactly one item in that set
                    if len(first_solution[nationality_row_idx][col]) == 1:
                        fish_owner_nationality = next(iter(first_solution[nationality_row_idx][col]))
                        fish_owner_house = col + 1
                    break
            
            # If we found a fish owner, add an explanatory log line
            if fish_owner_nationality:
                self._add_step(f"=> House#{fish_owner_house} owns the Fish ")
                # self._add_step(f"{fish_owner_house} lives the {fish_owner_nationality} ")
                self._add_step(f"=> the {fish_owner_nationality} owns the Fish ")
            else:
                self._add_step(f"No Solution ")
        else:
            self._add_step(f"No Solution ")

        return bool(solutions), solutions or [puzzle_ranges], fish_owner_nationality

    ##########################################
    # Main Algorithm
    ##########################################
    def _solve_puzzle(self, ranges: List[List[Set[str]]], constraints: Dict[str, Constraint]) -> List[List[List[Set[str]]]]:
        """
        Single-pass constraint propagation (with dictionary constraints),
        remove constraints that are “satisfied,” 
        then if not solved => branching.
        """
        # decide cat_names
        if self.num_categories == 5:
            cat_names = CATEGORY_NAMES_5
        elif self.num_categories == 4:
            cat_names = CATEGORY_NAMES_4
        elif self.num_categories == 3:
            cat_names = CATEGORY_NAMES_3
        else:
            cat_names = [f"Cat_{i}" for i in range(self.num_categories)]

        # 1) Print puzzle state & constraints
        self._add_step("[CALL] ====== Possible Assignments ======")
        self._add_step(format_ranges_nl(ranges, cat_names))

        # Show constraints
        if not constraints:
            self._add_step("No constraints left")
        else:
            self._add_step("Unsatisfied constraints are " + ' '.join([f"Constraint#{int(cid)+1}" for cid in constraints.keys()]))


        # 2) Check if puzzle is solved (each cell has exactly 1 item).
        #    BUT we must also check that *all constraints* truly hold.
        if all(len(cell) == 1 for row in ranges for cell in row):
            # We do a final check: if all constraints are indeed satisfied => solution
            for cid, c in constraints.items():
                if not is_constraint_satisfied(ranges, c):
                    # If even one constraint fails => contradiction
                    self._add_step("[SEP] No Solution [RETURN]")
                    return []

            # If we pass the loop => all constraints are satisfied => real solution
            self._add_step("=> Puzzle is solved")
            self._add_step('[SEP] Solution '+ format_ranges_nl(ranges, cat_names) + ' [RETURN]')
            return [copy.deepcopy(ranges)]
        
        self._add_step("=> Puzzle not solved yet")

        # 3) Single-pass constraint propagation
        self._add_step("====== Propagation ======")
        changed = True
        max_iterations = 1
        iteration = 0

        # We'll copy constraints to avoid messing original dictionary
        # We'll do one pass => for each constraint => _update_range
        # Then we check if the constraint is now satisfied => remove from next recursion if so
        # We'll store newly updated constraints in a separate dict
        new_constraints = copy.deepcopy(constraints)

        while changed and iteration < max_iterations:
            changed = False
            # We'll track which constraints become "satisfied"
            satisfied_ids = []
            for cid, c in list(new_constraints.items()):
                old_snapshot = copy.deepcopy(ranges)

                # self._add_step(f"=> Applying Constraint#{int(cid)+1} : {c.debug_text} , relation = {c.relation_name}")
                self._add_step(f"Applying Constraint#{int(cid)+1} [CALL]")
                # 3A) apply
                updated = self._update_range(c.values,
                                             [ranges[i] for i in c.attr_indices],
                                             c.compare,
                                             cat_names,
                                             c.relation_name,
                                             c.attr_indices)

                # 3B) show diff
                diffs = _print_diff(old_snapshot, ranges, cat_names)
                if diffs:
                    # self._add_step("Changes from this constraint:")
                    diffs[0] = "[SEP] " + diffs[0]
                    diffs[-1] += " [RETURN]"
                    for dline in diffs:
                        self._add_step(dline)
                else:
                    self._add_step("[SEP] No changes from this constraint [RETURN]")

                # 3C) check if this constraint is satisfied
                if is_constraint_satisfied(ranges, c):
                    self._add_step(f"Remove Constraint#{int(cid)+1} because it is satisfied")
                    satisfied_ids.append(cid)
                    
                if updated:
                    changed = True
                    # check contradiction
                    if any(not cell for row_cat in ranges for cell in row_cat):
                        self._add_step('[SEP] No Solution [RETURN]')
                        return []

                iteration += 1

            # remove satisfied from new_constraints
            for sid in satisfied_ids:
                del new_constraints[sid]
                
        self._add_step("[SEP] [CALL] ====== Possible Assignments After Propagation ======")
        self._add_step(format_ranges_nl(ranges, cat_names))
        if not constraints:
            self._add_step("No constraints left")
        else:
            self._add_step("Unsatisfied constraints are " + ' '.join([f"Constraint#{int(cid)+1}" for cid in new_constraints.keys()]))
        self.reasoning_steps[-1] += " [RETURN]"
        
        # 4) Check if puzzle is solved after single pass
        if all(len(cell) == 1 for row in ranges for cell in row):
            # We do a final check: if all constraints are indeed satisfied => solution
            for cid, c in constraints.items():
                if not is_constraint_satisfied(ranges, c):
                    # If even one constraint fails => contradiction
                    self._add_step("[SEP] No Solution [RETURN]")
                    return []

            # If we pass the loop => all constraints are satisfied => real solution
            self._add_step("=> Puzzle is solved")
            self._add_step('[SEP] Solution '+ format_ranges_nl(ranges, cat_names) + ' [RETURN]')
            return [copy.deepcopy(ranges)]
        
        self._add_step("=> Puzzle not solved yet")

        # 5) Branch
        self._add_step("====== Branch ======")
        
        min_size = float('inf')
        pick_row = 0
        pick_col = 0
        for i, row_cat in enumerate(ranges):
            for j, sset in enumerate(row_cat):
                if 1 < len(sset) < min_size:
                    min_size = len(sset)
                    pick_row, pick_col = i, j

        if min_size == float('inf'):
            # self._add_step("No moves left => contradiction => no solution")
            self._add_step('[SEP] No Solution [RETURN]')
            return []

        # do branching
        house_number = pick_col + 1
        category_label = cat_names[pick_row]
        possibilities_list = sorted(ranges[pick_row][pick_col])
        self._add_step(
            f"Branching on House#{house_number} {category_label} category with {min_size} possibilities {' '.join(possibilities_list)}"
        )        
        
        # 4) We'll try each possibility in turn
        backup = copy.deepcopy(ranges)
        used_values = set()
        # optionally skip singletons in the same category row
        for cset in ranges[pick_row]:
            if len(cset) == 1:
                used_values |= cset

        # The actual possibilities are anything in `cell_set - used_values`
        # if you want to skip re-used singletons in that row:
        cell_set = ranges[pick_row][pick_col]
        candidates = cell_set - used_values

        for val in sorted(candidates):
            self._add_step(
                f"Trying possibility {val} in House#{house_number} {category_label} category"
            )
            ranges[pick_row][pick_col] = {val}

            # 5B) Recur => pass new_constraints
            sub_solutions = self._solve_puzzle(ranges, new_constraints)
            if sub_solutions:
                # self._add_step(f"=> That assumption yields {len(sub_solutions)} solutions")
                self._add_step('[SEP] Solution '+ format_ranges_nl(sub_solutions[0], cat_names) + ' [RETURN]')
                return sub_solutions
            else:
                # self._add_step("=> Contradiction, no solution for this branch")
                pass

            ranges = copy.deepcopy(backup)
        
        self._add_step("[SEP] No Solution [RETURN]")
        return []


    def _update_range(self, wns: List[str], rns: List[List[Set[str]]], cmp_func: Callable, cat_names: List[str], relation_name: str, global_indices: List[int]) -> bool:
        """
        Single pass for one constraint => returns True if puzzle changed, else False.

        PHASE 1 (Singleton Logic):
        - Remove pinned items from other cells in the same row.
        - If an item can only fit in one cell => force it.

        PHASE 2 (Handle LEFT/RIGHT/SAME relations):
        - If 'A LEFT B', remove 'A' from rightmost house, remove 'B' from leftmost, etc.
        - If pinned => we remove conflicting houses and possibly force the other item next door.
        - If 'SAME', unify them in the same house.
        """
        changed = False

        # Shortcut references
        itemA, itemB = wns
        rowA, rowB = rns
        rowIdxA, rowIdxB = global_indices
        catA = cat_names[rowIdxA]
        catB = cat_names[rowIdxB]
        
        ########################################
        # PHASE 1: Singleton Logic
        ########################################
        self._add_step(f"PHASE 1: Single-value logic for {itemA} and {itemB} under {relation_name} constraint")
        # self._add_step(f"PHASE 1: Single-value logic")

        # For each row in rns (which usually is row for itemA and row for itemB):
        for row_idx, row in zip([rowIdxA, rowIdxB], rns):
            cat_label = cat_names[row_idx]

            # 1) gather pinned items (singletons)
            pinned_items = set()
            for s in row:
                if len(s) == 1:
                    pinned_items |= s

            # 2) remove pinned items from other sets in that row
            for house_i, cell in enumerate(row):
                if len(cell) != 1:
                    old_cell = cell.copy()
                    cell.difference_update(pinned_items)
                    removed = old_cell - cell
                    for itm in removed:
                        self._add_step(
                            f"Removing {itm} from House#{house_i+1} {cat_label} category because {itm} is pinned in another house"
                        )
                    if removed:
                        changed = True

            # 3) if an item can only go in one house => force it
            #    track possible columns for each item
            item_pos_map = {}
            for house_i, cell in enumerate(row):
                for itm in cell:
                    item_pos_map.setdefault(itm, set()).add(house_i)

            for itm, possible_cols in item_pos_map.items():
                if len(possible_cols) == 1:
                    unique_col = next(iter(possible_cols))
                    if len(row[unique_col]) > 1:
                        row[unique_col].clear()
                        row[unique_col].add(itm)
                        self._add_step(
                            f"Forcing {itm} in House#{unique_col+1} {cat_label} category because it can only appear here"
                        )
                        changed = True

        ########################################
        # PHASE 2: LEFT/RIGHT/SAME relations
        ########################################
        self._add_step(f"PHASE 2: Handling relation {itemA} {relation_name} {itemB}")

        def remove_item(rowset: List[Set[str]], house_i: int, itm: str, explanation: str):
            """Helper to remove 'itm' from rowset[house_i], with a short reason"""
            if itm in rowset[house_i]:
                rowset[house_i].remove(itm)
                self._add_step(explanation)
                nonlocal changed
                changed = True

        def force_item(rowset: List[Set[str]], house_i: int, itm: str, cat_label: str, explanation: str):
            """Helper to 'force' itm in rowset[house_i]"""
            if len(rowset[house_i]) > 1:
                rowset[house_i].clear()
                rowset[house_i].add(itm)
                self._add_step(explanation)
                nonlocal changed
                changed = True

        if relation_name == "LEFT":
            # A must be IMMEDIATELY to the LEFT of B
            self._add_step(f"{itemA} is immediately LEFT of {itemB}")

            # 1) Remove itemA from the right-most house
            rm_text = f"Removing {itemA} from House#{self.num_houses} because {itemA} can't be in the rightmost house if it's to the LEFT of {itemB}"
            remove_item(rowA, self.num_houses - 1, itemA, rm_text)

            # 2) Remove itemB from the left-most house
            rm_text = f"Removing {itemB} from House#1 because {itemB} can't be in the leftmost house if it's to the RIGHT of {itemA}"
            remove_item(rowB, 0, itemB, rm_text)

            # If itemA pinned to i => itemB must be i+1
            pinnedA = [i for i in range(self.num_houses) if rowA[i] == {itemA}]
            for i in pinnedA:
                # remove B from houses <= i
                for hh in range(i + 1):
                    remove_item(rowB, hh, itemB,
                        f"Since {itemA} is pinned to House#{i+1} , removing {itemB} from House#{hh+1} because {itemB} must be right of House#{i+1}")
                # remove B from houses >= i+2
                for hh in range(i + 2, self.num_houses):
                    remove_item(rowB, hh, itemB,
                        f"{itemB} must be immediately to the right => removing from House#{hh+1}")
                # force B in i+1 if valid
                if i + 1 < self.num_houses:
                    force_item(rowB, i + 1, itemB, catB,
                        f"Placing {itemB} in House#{i+2} because {itemA} is pinned to House#{i+1}")

            # If itemB pinned => itemA must be pinned one house left
            pinnedB = [j for j in range(self.num_houses) if rowB[j] == {itemB}]
            for j in pinnedB:
                # remove A from houses >= j
                for hh in range(j, self.num_houses):
                    remove_item(rowA, hh, itemA,
                        f"Since {itemB} is pinned to House#{j+1} , {itemA} must be left => removing from House#{hh+1}")
                # remove A from houses < j-1
                for hh in range(0, j - 1):
                    remove_item(rowA, hh, itemA,
                        f"{itemA} must be exactly one left => removing from House#{hh+1}")
                if j - 1 >= 0:
                    force_item(rowA, j - 1, itemA, catA,
                        f"Placing {itemA} in House#{j} because {itemB} is pinned to House#{j+1}")

        elif relation_name == "RIGHT":
            # A must be IMMEDIATELY to the RIGHT of B
            self._add_step(f"{itemA} is immediately RIGHT of {itemB}")

            # 1) Remove itemA from the left-most house
            rm_text = f"Removing {itemA} from House#1 because {itemA} can't be in the leftmost house if it's to the RIGHT of {itemB}"
            remove_item(rowA, 0, itemA, rm_text)

            # 2) Remove itemB from the right-most house
            rm_text = f"Removing {itemB} from House#{self.num_houses} can't be in the rightmost house if it's to the LEFT of {itemA}"
            remove_item(rowB, self.num_houses - 1, itemB, rm_text)

            # If itemA pinned to i => itemB must be i-1
            pinnedA = [i for i in range(self.num_houses) if rowA[i] == {itemA}]
            for i in pinnedA:
                left_house = i - 1
                # # remove B from houses >= i
                for hh in range(i, self.num_houses):
                    remove_item(rowB, hh, itemB,
                        f"Since {itemA} is pinned to House#{i+1} , removing {itemB} from House#{hh+1} because {itemB} must be left of House#{i+1}")
                # remove B from houses < i-1
                for hh in range(0, i - 1):
                    remove_item(rowB, hh, itemB,
                        f"{itemB} must be exactly one house to the LEFT , removing from House#{hh+1}")
                if left_house >= 0:
                    force_item(rowB, left_house, itemB, catB,
                        f"Placing {itemB} in House#{left_house+1} because {itemA} is pinned to House#{i+1}")

            # If itemB pinned => itemA must be pinned one house to the right
            pinnedB = [j for j in range(self.num_houses) if rowB[j] == {itemB}]
            for j in pinnedB:
                right_house = j + 1
                # remove A from houses <= j
                for hh in range(0, j + 1):
                    remove_item(rowA, hh, itemA,
                        f"Since {itemB} is pinned to House#{j+1} , removing {itemA} from House#{hh+1} because {itemA} must be right of House#{j+1}")
                # remove A from houses > j+1
                for hh in range(j + 2, self.num_houses):
                    remove_item(rowA, hh, itemA,
                        f"{itemA} must be exactly one house to the RIGHT , removing from House#{hh+1}")
                if right_house < self.num_houses:
                    force_item(rowA, right_house, itemA, catA,
                        f"Placing {itemA} in House#{right_house+1} because {itemB} is pinned to House#{j+1}")
        
        elif relation_name == "SAME":
            self._add_step(f"{itemA} must be in the SAME house as {itemB}")
            pinnedA = [i for i in range(self.num_houses) if rowA[i] == {itemA}]
            pinnedB = [j for j in range(self.num_houses) if rowB[j] == {itemB}]

            # if A pinned => B must be forced there
            for i in pinnedA:
                force_item(rowB, i, itemB, catB,
                    f"Placing {itemB} in House#{i+1} since {itemA} is in this house")
                for col_rem in range(self.num_houses):
                    if col_rem != i:
                        remove_item(rowB, col_rem, itemB,
                            f"Since {itemA} is pinned to House#{i+1} , removing {itemB} from House#{col_rem+1}")


            # if B pinned => A must be forced
            for j in pinnedB:
                force_item(rowA, j, itemA, catA,
                    f"Placing {itemA} in House#{j+1} since {itemB} is in this house")
                for col_rem in range(self.num_houses):
                    if col_rem != j:
                        remove_item(rowA, col_rem, itemA,
                            f"Since {itemB} is pinned to House#{j+1} , removing {itemA} from House#{col_rem+1}")


            # also remove itemA from any house that can't hold itemB, and vice versa
            for col in range(self.num_houses):
                if itemB not in rowB[col] and itemA in rowA[col]:
                    remove_item(rowA, col, itemA,
                        f"House#{col+1} can't hold {itemB} since it can't hold {itemA}")
                if itemA not in rowA[col] and itemB in rowB[col]:
                    remove_item(rowB, col, itemB,
                        f"House#{col+1} can't hold {itemA} since it can't hold {itemB}")

        else:
            self._add_step(f"(Skipped) Relation {relation_name} not recognized for direct approach")

        return changed
    
def solve_dataset(data_file: str):
    if not os.path.exists(data_file):
        print(f"File {data_file} not found")
        return

    # Read all data first
    all_data = []
    with open(data_file, "r", encoding="utf-8") as f:
        all_data = [json.loads(line.strip()) for line in f]

    # Process and update data
    with open(data_file, "w", encoding="utf-8") as f:
        for puzzle_idx, data in enumerate(all_data, 1):
            size = data["size"]
            table = PUZZLES[size]

            constraints_dict = {}
            for c_info in data["constraints"]:
                cmp_func, rel_name = make_constraint_function(c_info["relation"], len(table[0]))
                c_obj = Constraint(
                    attr_indices=c_info["attr_indices"],
                    values=c_info["item_values"],
                    compare=cmp_func,
                    debug_text=c_info["debug_text"],
                    relation_name=rel_name
                )
                c_id = str(len(constraints_dict))
                constraints_dict[c_id] = c_obj

            solver = EinsteinSolver(table)
            t0 = time.perf_counter()
            is_solved, solutions, fish_owner = solver.solve(constraints_dict)
            dt = time.perf_counter() - t0
            
            # Update text field with reasoning steps
            data['text'] = ' \n '.join(solver.reasoning_steps)
            f.write(json.dumps(data) + '\n')
            
            if fish_owner == data.get("label", "???"):
                # print(f"Puzzle #{puzzle_idx} solved correctly")
                pass
            else:
                print(f"Puzzle #{puzzle_idx} solved incorrectly")
                print(f"Expected fish owner: {data.get('label', '???')}, Got: {fish_owner}")
                print(solutions)
                print(data['solution'])

            # Console output
            # print(f"\n====== Puzzle #{puzzle_idx} ======")
            # print(f"Size: {size}, Label: {data.get('label', '???')}")
            # print(f"Solved: {is_solved}, Solutions: {len(solutions)}, Time: {dt:.4f}s")

            if is_solved:
                for i, sol in enumerate(solutions[:2], 1):
                    # print(f"\n--- Solution {i} ---")
                    final_table = []
                    for row in sol:
                        final_table.append([
                            next(iter(cell)) if len(cell) == 1 else "???" for cell in row
                        ])
                    # print(format_table(final_table))
            else:
                # print("No valid solution or puzzle ended in contradiction")
                pass
            # print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Einstein Puzzle Solver (single pass, dictionary constraints, detailed logs)")
    parser.add_argument("--data_dir", default="data/einstein_3or5", help="Puzzle dataset JSONL")
    parser.add_argument("--train_size", type=int, default=10000, help="Number of puzzles to generate")
    args = parser.parse_args()
    args.data_file = os.path.join(args.data_dir, "data.jsonl")
    
    solve_dataset(args.data_file)
    
    # OPTIONAL: Build tokenizer, process dataset, etc.
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab_from_file(args.data_dir)
    tokenizer.save_meta_to_file(args.data_dir)
    
    process_dataset(args.data_dir, train_size=args.train_size, val_size=100, test_size=100, tokenizer=tokenizer)
    
    process_pencil_data(args.data_dir, tokenizer)