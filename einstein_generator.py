import random
import collections
import time
import json
import argparse
import os
from typing import List, Set, Tuple, Callable, Optional, Dict

###############################################################################
# 1) Domain definitions for 3×3 or 5×5
###############################################################################
COLORS_3  = ["Red", "Green", "Blue"]
NATION_3  = ["Brit", "Swede", "German"]
PET_3     = ["Dogs", "Birds", "Fish"]

COLORS_4  = ["Red", "Green", "Blue", "Yellow"]
NATION_4  = ["Brit", "Swede", "German", "Dane"]
DRINK_4   = ["Tea", "Coffee", "Milk", "Beer"]
PET_4     = ["Dogs", "Birds", "Fish", "Cats"]

COLORS_5  = ["Red", "Green", "White", "Yellow", "Blue"]
NATION_5  = ["Brit", "Swede", "Dane", "Norwegian", "German"]
DRINK_5   = ["Tea", "Coffee", "Milk", "Beer", "Water"]
SMOKE_5   = ["Pall Mall", "Dunhill", "Blend", "Blue Master", "Prince"]
PET_5     = ["Dogs", "Birds", "Cats", "Horses", "Fish"]

###############################################################################
# 2) Build default table => final arrangement
###############################################################################
def build_default_table(size=5):
    """
    If size=3 => 3 attributes => [Color, Nationality, Pet], each with 3 items => 3 houses
    If size=4 => 4 attributes => [Color, Nationality, Pet, Drink], each with 4 items => 4 houses
    If size=5 => 5 attributes => [Color, Nationality, Drink, Smoke, Pet], each with 5 items => 5 houses
    We'll shuffle each row, store as [heading, item1, item2, ...].
    """
    if size == 3:
        table = [
            ["Color"]       + random.sample(COLORS_3,   3),
            ["Nationality"] + random.sample(NATION_3,   3),
            ["Pet"]         + random.sample(PET_3,      3),
        ]
    elif size == 4:
        table = [
            ["Color"]       + random.sample(COLORS_4,   4),
            ["Nationality"] + random.sample(NATION_4,   4),
            ["Drink"]       + random.sample(DRINK_4,    4),
            ["Pet"]         + random.sample(PET_4,      4),
        ]
    else:
        # size=5
        table = [
            ["Color"]       + random.sample(COLORS_5,   5),
            ["Nationality"] + random.sample(NATION_5,   5),
            ["Drink"]       + random.sample(DRINK_5,    5),
            ["Smoke"]       + random.sample(SMOKE_5,    5),
            ["Pet"]         + random.sample(PET_5,      5)
        ]
    return table

###############################################################################
# 3) Make short English phrases for debug_text
###############################################################################
def make_natural(category: str, value: str) -> str:
    if category == "Color":
        return f"the {value} house"
    elif category == "Nationality":
        return f"the {value}"
    elif category == "Drink":
        return f"the one who drinks {value}"
    elif category == "Smoke":
        return f"the one who smokes {value}"
    elif category == "Pet":
        return f"the one who keeps {value}"
    return f"the {value}"

###############################################################################
def build_debug_text(rname: str, catA: str, valA: str, catB:str="", valB:str="") -> str:
    A = make_natural(catA, valA)
    B = make_natural(catB, valB) if catB else ""
    if rname=="same":
        return f"{A} is the same house as {B}"
    elif rname=="left":
        return f"{A} is immediately to the left of {B}"
    elif rname=="right":
        return f"{A} is immediately to the right of {B}"
    elif rname=="far_left":
        return f"{A} is on the far left"
    elif rname=="far_right":
        return f"{A} is on the far right"
    elif rname=="middle":
        return f"{A} is in the middle"
    return f"{A} ??? {B}"

###############################################################################
# 4) The partial BFS check used to see if there's more than one solution
#    in the final "minimization" step
###############################################################################
def update_ranges(relations, ranges):
    changed=False
    for (attr_idxs, item_vals, cmp_func, debug_txt) in relations:
        sub_rns = [ ranges[i] for i in attr_idxs ]
        c= _update_range(item_vals, sub_rns, cmp_func)
        if c:
            changed=True
    return changed

def _update_range(wns: List[str], rns: List[List[Set[str]]], cmp: Callable) -> bool:
    """
    The same 'update_range' code from your snippet. 
    """
    changed = False
    for rn in rns:
        classified_words = set()
        for n_col, set_of_words in enumerate(rn):
            if len(set_of_words) == 1:
                classified_words.add(next(iter(set_of_words)))
        word_to_cols = {}
        for n_col, set_of_words in enumerate(rn):
            if len(set_of_words) != 1:
                before = len(set_of_words)
                set_of_words.difference_update(classified_words)
                if len(set_of_words) != before:
                    changed = True
                for w in set_of_words:
                    word_to_cols.setdefault(w, set()).add(n_col)
        for w, cols in word_to_cols.items():
            if len(cols)==1:
                x= rn[next(iter(cols))]
                if len(x)!=1:
                    x.clear()
                    x.add(w)
                    changed=True

    new_rns = [[{x for x in xs if x != wn} for xs in rn] for wn,rn in zip(wns,rns)]
    pairs=[]
    for wn, rn in zip(wns, rns):
        new_pairs=[]
        break_condition=True
        for cn, setn in enumerate(rn):
            if wn in setn:
                break_condition=False
                if not pairs:
                    pairs=[[]]
                for v in pairs:
                    new_pairs.append([*v, cn])
        pairs=new_pairs
        if break_condition:
            break
    for pair in pairs:
        if cmp(*pair):
            for nrn,cn,ww in zip(new_rns, pair, wns):
                nrn[cn].add(ww)

    old_vs_new = any(rn!=nrn for rn, nrn in zip(rns,new_rns))
    changed |= old_vs_new
    if old_vs_new:
        for rn,nrn in zip(rns,new_rns):
            for old,new in zip(rn,nrn):
                old.intersection_update(new)
    return changed

def is_solved(ranges)->bool:
    for row_ in ranges:
        for cell_ in row_:
            if len(cell_)!=1:
                return False
    return True

def is_contradiction(ranges)->bool:
    for row_ in ranges:
        for cell_ in row_:
            if len(cell_)==0:
                return True
    return False

def copy_matrix(ranges):
    return [[x.copy() for x in row] for row in ranges]

###############################################################################
# 5) Build the final solution (houses) from table
###############################################################################
def build_solution_from_table(table: List[List[str]])->List[Dict[str,str]]:
    n_attr=len(table)
    m_obj=len(table[0])-1
    houses=[]
    for col_j in range(m_obj):
        house_data={"position": col_j+1}
        for row_i in range(n_attr):
            cat= table[row_i][0]
            val= table[row_i][col_j+1]
            house_data[cat]= val
        houses.append(house_data)
    return houses

def find_fish_owner(houses)->str:
    for h in houses:
        if h.get("Pet")=="Fish":
            return h.get("Nationality","Unknown")
    return "Unknown"

###############################################################################
# 6) The puzzle generator that does partial BFS for minimization.
#    We'll produce "constraints" in structured format, not plain text.
###############################################################################
def generate_puzzle(table: List[List[str]], 
                    level=20,
                    minimal_conditions=True,
                    max_seconds_for_minimizing=30.0,
                    tries=5)->List[Dict]:
    """
    - We store constraints in a structured way => "constraints"
      Each constraint => { "attr_indices", "item_values", "relation", "debug_text" }.
    - In the final BFS step, if it sees multiple solutions, it tries removing constraints.
    """
    table_wo_left= [ row[1:] for row in table ]
    n_attributes= len(table_wo_left)
    m_objects= len(table_wo_left[0])

    base_rules = [
        # (n_args, compare_func, [some template], relation_name)
        (2, lambda j1,j2: j1==j2,        ["{0} is the same house as {2}"],    "same"),
        (2, lambda j1,j2: j1==(j2-1),    ["{0} is immediately to the left of {2}"],  "left"),
        (2, lambda j1,j2: j1==(j2+1),    ["{0} is immediately to the right of {2}"], "right"),
        (1, lambda j1: j1==0,            ["{0} is on the far left"], "far_left"),
        (1, lambda j1,li=(m_objects-1): j1==li, ["{0} is on the far right"], "far_right")
    ]
    if m_objects%2!=0:
        center= m_objects//2
        base_rules.append((1,lambda j1,mid=center: j1==mid, ["{0} is in the middle"], "middle"))

    # We'll produce constraints in "structured" form => 
    #    { "attr_indices":[...], "item_values":[...], "relation":"...", "debug_text":"..." }

    min_relations= None
    time_elapsed= False
    is_minimized= False

    while True:
        # ranges => domain sets
        ranges= [[ set(row) for _ in range(m_objects)] for row in table_wo_left]
        relations=[]
        fail=False
        while not fail:
            needs_clarification=[]
            no_solutions=False
            solved=True
            for i,rng in enumerate(ranges):
                for j, cellset in enumerate(rng):
                    if len(cellset)==0:
                        no_solutions=True
                        solved=False
                        break
                    elif len(cellset)>1:
                        solved=False
                        needs_clarification.append((i,j))
                if no_solutions:
                    break

            if solved or (min_relations is not None and len(relations)>= len(min_relations)):
                tries-=1
                if min_relations is None or len(relations)< len(min_relations):
                    min_relations= relations
                if tries>0:
                    fail=True
                    continue

            if tries<=0:
                # finalize
                relations= min_relations
                if not minimal_conditions:
                    break
                # Minimization BFS => partial BFS check
                number_of_relations_min= len(relations)
                number_of_relations_before= len(relations)
                start_time= time.monotonic()
                main_q= collections.deque([ relations ])
                while main_q:
                    current_relations= main_q.popleft()
                    for k in range(len(current_relations)):
                        new_ranges=[[ set(row) for row in table_wo_left[i]] for i in range(len(table_wo_left))]
                        # Actually we need correct shape:
                        new_ranges = [[ set(table_wo_left[i]) for _ in range(len(table_wo_left[i]))]
                                      for i in range(len(table_wo_left))]
                        new_relations= current_relations.copy()
                        new_relations.pop(k)
                        changed=True
                        while changed:
                            changed= update_ranges(new_relations, new_ranges)

                        # BFS check
                        q= collections.deque([new_ranges])
                        possible_solutions=[]
                        while q:
                            cur_ranges= q.popleft()
                            no_sol=False
                            solved_inner=True
                            for row_ in cur_ranges:
                                for cell_ in row_:
                                    if len(cell_)==0:
                                        no_sol=True
                                        solved_inner=False
                                        break
                                    elif len(cell_)>1:
                                        solved_inner=False
                                if no_sol or not solved_inner:
                                    break
                            if no_sol:
                                continue
                            if solved_inner:
                                if cur_ranges not in possible_solutions:
                                    possible_solutions.append(cur_ranges)
                                    if len(possible_solutions)>=2:
                                        break
                                continue
                            # expand
                            branched=False
                            for rr_i, row_ in enumerate(cur_ranges):
                                for cc_j, cell_ in enumerate(row_):
                                    if len(cell_)>1:
                                        branched=True
                                        for val in list(cell_):
                                            cpy= copy_matrix(cur_ranges)
                                            cpy[rr_i][cc_j] = {val}
                                            changed2=True
                                            while changed2:
                                                changed2= update_ranges(new_relations, cpy)
                                            q.appendleft(cpy)
                                        break
                                if branched:
                                    break

                        if len(possible_solutions)==1:
                            number_of_relations_after= len(new_relations)
                            if number_of_relations_min> number_of_relations_after:
                                number_of_relations_min= number_of_relations_after
                                relations= new_relations
                                main_q.append(new_relations)

                        if (max_seconds_for_minimizing is not None and 
                            time.monotonic()>= (start_time+max_seconds_for_minimizing)):
                            time_elapsed=True
                            break
                    if time_elapsed:
                        break
                is_minimized= (number_of_relations_min< number_of_relations_before) or not time_elapsed
                break

            if no_solutions or not needs_clarification:
                fail=True
                continue

            # pick random cell
            i,j= random.choice(needs_clarification)
            neighbors=[]
            for dj in [-1,0,1]:
                jj= j+dj
                if 0<=jj<m_objects:
                    for new_i in range(n_attributes):
                        if new_i==i and dj==0:
                            continue
                        neighbors.append((new_i,jj))
            if not neighbors:
                continue
            next_i,next_j= random.choice(neighbors)

            # pick random rule => only handle n_args=1 or 2
            possible_variants=[]
            for (n_args,cmp_func, templates,rel_name) in base_rules:
                if n_args==1 and i==next_i and j== next_j:
                    if cmp_func(j):
                        possible_variants.append((1,[(i,j)],cmp_func, random.choice(templates),rel_name))
                elif n_args==2 and (i!=next_i or j!= next_j):
                    if cmp_func(j,next_j):
                        possible_variants.append((2,[(i,j),(next_i,next_j)],cmp_func, random.choice(templates),rel_name))

            if not possible_variants:
                continue
            chosen= random.choice(possible_variants)
            n_args, list_of_ij, cmp_function, template, rel_name= chosen
            ins=[]
            wns=[]
            debug_text= None
            if n_args==2:
                (attr1,col1),(attr2,col2)= list_of_ij
                cat1= table[attr1][0]
                val1= table_wo_left[attr1][col1]
                cat2= table[attr2][0]
                val2= table_wo_left[attr2][col2]
                debug_text= template.format(
                    make_natural(cat1,val1),
                    "", 
                    make_natural(cat2,val2),
                    ""
                )
                ins=[attr1,attr2]
                wns=[val1,val2]
                relations.append((ins, wns, cmp_function, debug_text))
            else:
                (attrx,colx)= list_of_ij[0]
                catx= table[attrx][0]
                valx= table_wo_left[attrx][colx]
                debug_text= template.format( make_natural(catx,valx) )
                ins=[attrx]
                wns=[valx]
                relations.append((ins, wns, cmp_function, debug_text))

            changed=True
            while changed:
                changed= update_ranges(relations, ranges)

        if not fail:
            break

    # We produce a structured constraints list
    structured_constraints=[]
    if min_relations:
        for (attr_idxs, item_vals, cmp_func, dbg) in min_relations:
            # we guess relation from dbg if you want. 
            # We'll parse the text. Or store 'rel_name' above. 
            # We'll do a small parse:
            rel="unknown"
            if "the same house" in dbg:
                rel="same"
            elif "to the left of" in dbg:
                rel="left"
            elif "to the right of" in dbg:
                rel="right"
            elif "on the far left" in dbg:
                rel="far_left"
            elif "on the far right" in dbg:
                rel="far_right"
            elif "in the middle" in dbg:
                rel="middle"
            # else "unknown"

            cdict={
              "attr_indices": attr_idxs,
              "item_values": item_vals,
              "relation": rel,
              "debug_text": dbg
            }
            structured_constraints.append(cdict)

    random.shuffle(structured_constraints)
    return structured_constraints

###############################################################################
# 7) Build puzzle: structured constraints + solution + label + text
###############################################################################
def build_text_from_constraints(constraints: List[Dict], fish_owner:str)->str:
    lines=[]
    for i,c in enumerate(constraints, start=1):
        lines.append(f"Clue{i}: {c['debug_text']}")
    text= "\n".join(lines)
    text+= f" <|endofprompt|> {fish_owner} keeps fish"
    return text

###############################################################################
# 8) The main generator class
###############################################################################
class EinsteinPuzzleGenerator:
    def __init__(self, size=5, minimal_conditions=True, max_seconds_for_minimizing=10.0, tries=5):
        # Now accepts 3, 4, or 5
        if size not in [3, 4, 5]:
            raise ValueError("Size must be 3, 4, or 5.")
        self.size = size
        self.minimal_conditions = minimal_conditions
        self.max_seconds_for_minimizing = max_seconds_for_minimizing
        self.tries = tries

    def generate_dataset(
        self,
        num_samples: int,
        data_dir: Optional[str] = None,
        save: bool = True,
        batch_size: int = 10000,
        buffer_size: int = 1000
    ):
        if save and not data_dir:
            raise ValueError("Must provide data_dir if save=True")

        data = [] if not save else None
        file_handle= None
        buffer_list= []

        try:
            if save:
                os.makedirs(data_dir, exist_ok=True)
                outfile= os.path.join(data_dir, "data.jsonl")
                file_handle= open(outfile,"w", encoding="utf-8")

            for i in range(num_samples):
                if (i+1) % 10==0:
                    print(f"Generating puzzle {i+1}/{num_samples}")
                puzzle= self.generate_instance()
                if save:
                    buffer_list.append(json.dumps(puzzle))
                    if len(buffer_list)>= buffer_size:
                        file_handle.write("\n".join(buffer_list)+"\n")
                        buffer_list.clear()
                else:
                    data.append(puzzle)

            if save and buffer_list:
                file_handle.write("\n".join(buffer_list)+"\n")
                buffer_list.clear()
        finally:
            if file_handle:
                file_handle.close()

        return data

    def generate_instance(self)->Dict:
        """
        1) Build random table => final arrangement
        2) generate puzzle => structured constraints
        3) build solution => label => text
        """
        table= build_default_table(self.size)
        # generate structured constraints
        constraints= generate_puzzle(
            table,
            level=20,
            minimal_conditions=self.minimal_conditions,
            max_seconds_for_minimizing=self.max_seconds_for_minimizing,
            tries=self.tries
        )
        solution= build_solution_from_table(table)
        fish_owner= find_fish_owner(solution)
        text_str= build_text_from_constraints(constraints, fish_owner)
        puzzle_item={
          "text": text_str,
          "constraints": constraints,
          "solution": solution,
          "label": fish_owner,
          "size": self.size
        }
        return puzzle_item

def main():
    parser = argparse.ArgumentParser("Einstein Puzzle Data Generator (3x3, 4x4, or 5x5) with structured constraints")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of puzzles to generate")
    parser.add_argument("--data_dir", type=str, default="data/einstein_3_4_5", help="Output dir")
    parser.add_argument("--size", type=int, default=3, choices=[3, 4, 5], help="Puzzle size => 3, 4, or 5")
    parser.add_argument("--minimal_conditions", action="store_true", help="Try removing extra clues")
    parser.add_argument("--max_seconds_for_minimizing", type=float, default=10.0, help="Time limit for minimization BFS")
    parser.add_argument("--tries", type=int, default=5, help="Number of attempts building constraints")
    parser.add_argument("--save", action="store_true", help="If set => store to data.jsonl")

    args = parser.parse_args()

    gen = EinsteinPuzzleGenerator(
        size=args.size,
        minimal_conditions=args.minimal_conditions,
        max_seconds_for_minimizing=args.max_seconds_for_minimizing,
        tries=args.tries
    )

    if args.save:
        os.makedirs(args.data_dir, exist_ok=True)

    dataset = gen.generate_dataset(
        num_samples=args.num_samples,
        data_dir=args.data_dir,
        save=args.save
    )

    if (not args.save) and dataset is not None:
        for i, pz in enumerate(dataset, start=1):
            print(f"\n--- Puzzle #{i} ---")
            print("text:\n", pz["text"])
            print("\nConstraints:")
            for c in pz["constraints"]:
                print(" ", c)
            print("\nSolution:")
            for h in pz["solution"]:
                print(" ", h)
            print("label:", pz["label"])
            print("size:", pz["size"])

if __name__ == "__main__":
    main()