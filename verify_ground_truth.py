import json
import re

# Samples with suspected incorrect ground truth
suspicious_samples = {
    "47": "(MED (SM (MAX (SM (SUM (AVG (SUM (MIN 4 2 15 15) 6 11 23 15 14 17 28) 30 29 3 22) 23 7 3 5 20) 23 25 1 19) 8 16 27 21 11 11) 10 2 28 6 27) 21 5 3 28 4 27)",
    "20": "(SM (SM (MIN (MIN (MIN (SUM (SUM (SUM 23 23 18 19 21 14 3 22) 4 6 9 3) 8 5 23 22) 11 14 22 26 3 19 3) 19 17 18 24 21) 13 27 2 22) 11 22 18 15 25 1) 6 22 29 15 19)",
    "37": "(AVG (MED (SM (SM (SM (SM (MIN (SM 29 30 27 3) 23 19 25 10) 29 25 11 21) 27 27 24 4) 6 21 9) 20 30 21 25 20 25 18) 3 5 26 18) 3 12 3)",
    "95": "(MED (SM (MAX (MIN (AVG (SUM (SUM (AVG 13 3 29 25 16 21 12) 22 29 10) 10 20 7 23 22 16) 20 1 16 7 6 22) 23 14 24 7 16 24) 5 28 1 11) 19 6 18 2 2 5 9) 22 2 19 3)"
}

# Ground truth values from the dataset
ground_truth_values = {
    "47": 5,
    "20": 5, 
    "37": 5,
    "95": 9
}

# Suggested answers from our verification
suggested_answers = {
    "47": 21,
    "20": 185,
    "37": 9,
    "95": 19
}

def eval_listops_expression(expr_str):
    """
    A more robust parser for ListOps expressions.
    """
    # Tokenize the expression
    tokens = re.findall(r'\(|\)|SUM|SM|MIN|MAX|MED|AVG|\d+', expr_str)
    
    def parse_expr(tokens, idx=0):
        if tokens[idx] != '(':
            # Must be a number
            return int(tokens[idx]), idx + 1
        
        # Get operator
        operator = tokens[idx + 1]
        idx += 2  # Skip '(' and operator
        
        # Parse operands
        operands = []
        while idx < len(tokens) and tokens[idx] != ')':
            if tokens[idx] == '(':
                val, next_idx = parse_expr(tokens, idx)
                operands.append(val)
                idx = next_idx
            elif tokens[idx].isdigit():
                operands.append(int(tokens[idx]))
                idx += 1
            else:
                idx += 1
        
        # Evaluate based on operator
        result = None
        if operator == "SUM":
            result = sum(operands)
        elif operator == "SM":
            result = sum(operands) % 10  # SM is sum modulo 10
        elif operator == "AVG":
            result = sum(operands) // len(operands)  # Integer division (floor)
        elif operator == "MIN":
            result = min(operands)
        elif operator == "MAX":
            result = max(operands)
        elif operator == "MED":
            sorted_ops = sorted(operands)
            if len(sorted_ops) % 2 == 1:
                # Odd length, take middle element
                result = sorted_ops[len(sorted_ops) // 2]
            else:
                # Even length, take average of two middle elements and floor it
                mid1 = sorted_ops[len(sorted_ops) // 2 - 1]
                mid2 = sorted_ops[len(sorted_ops) // 2]
                result = (mid1 + mid2) // 2
        
        return result, idx + 1  # Skip ')'
    
    result, _ = parse_expr(tokens)
    return result

def step_by_step_evaluation(expr):
    """
    Evaluates the expression step by step and prints the intermediate results.
    """
    # Implement this for a more detailed debugging trace
    # We'll use a simplified approach using different operations directly
    
    # Start with innermost expressions
    def find_innermost_expr(expr):
        # Find all expressions with no nested parentheses
        matches = re.finditer(r'\([^()]+\)', expr)
        results = []
        for match in matches:
            start, end = match.span()
            expr_part = expr[start:end]
            results.append((start, end, expr_part))
        return results
    
    print("\nStep-by-step evaluation:")
    current_expr = expr
    step = 1
    
    while '(' in current_expr:
        innermost = find_innermost_expr(current_expr)
        if not innermost:
            break
            
        for start, end, expr_part in innermost:
            try:
                # Parse the operator and operands
                match = re.match(r'\((\w+)(.*)\)', expr_part)
                if match:
                    op = match.group(1)
                    operands_str = match.group(2).strip()
                    operands = [int(x) for x in operands_str.split() if x.isdigit()]
                    
                    result = None
                    if op == "SUM":
                        result = sum(operands)
                        print(f"Step {step}: {expr_part} = {'+'.join(map(str, operands))} = {result}")
                    elif op == "SM":
                        result = sum(operands) % 10
                        print(f"Step {step}: {expr_part} = ({'+'.join(map(str, operands))}) % 10 = {result}")
                    elif op == "AVG":
                        result = sum(operands) // len(operands)
                        print(f"Step {step}: {expr_part} = ({'+'.join(map(str, operands))}) / {len(operands)} = {result}")
                    elif op == "MIN":
                        result = min(operands)
                        print(f"Step {step}: {expr_part} = min({', '.join(map(str, operands))}) = {result}")
                    elif op == "MAX":
                        result = max(operands)
                        print(f"Step {step}: {expr_part} = max({', '.join(map(str, operands))}) = {result}")
                    elif op == "MED":
                        sorted_ops = sorted(operands)
                        if len(sorted_ops) % 2 == 1:
                            result = sorted_ops[len(sorted_ops) // 2]
                            print(f"Step {step}: {expr_part} = median({', '.join(map(str, operands))}) = {result}")
                        else:
                            mid1 = sorted_ops[len(sorted_ops) // 2 - 1]
                            mid2 = sorted_ops[len(sorted_ops) // 2]
                            result = (mid1 + mid2) // 2
                            print(f"Step {step}: {expr_part} = median({', '.join(map(str, operands))}) = ({mid1}+{mid2})/2 = {result}")
                    
                    if result is not None:
                        # Replace the evaluated expression with its result
                        current_expr = current_expr[:start] + str(result) + current_expr[end:]
                        step += 1
                        break  # Start over with the new expression
            except Exception as e:
                print(f"Error evaluating {expr_part}: {e}")
                current_expr = current_expr[:start] + "ERROR" + current_expr[end:]
                break
    
    print(f"Final result: {current_expr}")
    return current_expr

def evaluate_and_print_steps(expr, sample_id):
    """Evaluate the expression and print detailed steps for debugging"""
    print(f"\nSample {sample_id}:")
    print(f"Expression: {expr}")
    print(f"Ground Truth: {ground_truth_values.get(sample_id)}")
    print(f"Suggested Answer: {suggested_answers.get(sample_id)}")
    
    try:
        # Manual calculation for simple verification
        if sample_id == "47":
            print("\nManual calculation for sample 47:")
            print("1. (MIN 4 2 15 15) = 2")
            print("2. (SUM 2 6 11 23 15 14 17 28) = 116")
            print("3. (AVG 116 30 29 3 22) = (116+30+29+3+22)/5 = 200/5 = 40")
            print("4. (SUM 40 23 7 3 5 20) = 98")
            print("5. (SM 98 23 25 1 19) = (98+23+25+1+19) % 10 = 166 % 10 = 6")
            print("6. (MAX 6 8 16 27 21 11 11) = 27")
            print("7. (SM 27 10 2 28 6 27) = (27+10+2+28+6+27) % 10 = 100 % 10 = 0")
            print("8. (MED 0 21 5 3 28 4 27) = median of [0,3,4,5,21,27,28] = 5")
        
        elif sample_id == "20":
            print("\nManual calculation for sample 20:")
            print("1. (SUM 23 23 18 19 21 14 3 22) = 143")
            print("2. (SUM 143 4 6 9 3) = 165")
            print("3. (SUM 165 8 5 23 22) = 223")
            print("4. (MIN 223 11 14 22 26 3 19 3) = 3")
            print("5. (MIN 3 19 17 18 24 21) = 3")
            print("6. (MIN 3 13 27 2 22) = 2")
            print("7. (SM 2 11 22 18 15 25 1) = (2+11+22+18+15+25+1) % 10 = 94 % 10 = 4")
            print("8. (SM 4 6 22 29 15 19) = (4+6+22+29+15+19) % 10 = 95 % 10 = 5")

        elif sample_id == "37":
            print("\nManual calculation for sample 37:")
            print("1. (SM 29 30 27 3) = (29+30+27+3) % 10 = 89 % 10 = 9")
            print("2. (MIN 9 23 19 25 10) = 9")
            print("3. (SM 9 29 25 11 21) = (9+29+25+11+21) % 10 = 95 % 10 = 5")
            print("4. (SM 5 27 27 24 4) = (5+27+27+24+4) % 10 = 87 % 10 = 7")
            print("5. (SM 7 6 21 9) = (7+6+21+9) % 10 = 43 % 10 = 3")
            print("6. (MED 3 20 30 21 25 20 25 18) = median of [3,18,20,20,21,25,25,30] = (20+21)/2 = 20")
            print("7. (AVG 20 3 5 26 18) = (20+3+5+26+18)/5 = 72/5 = 14")
            print("8. (AVG 14 3 12 3) = (14+3+12+3)/4 = 32/4 = 8")
            
        elif sample_id == "95":
            print("\nManual calculation for sample 95:")
            print("1. (AVG 13 3 29 25 16 21 12) = (13+3+29+25+16+21+12)/7 = 119/7 = 17")
            print("2. (SUM 17 22 29 10) = 78")
            print("3. (SUM 78 10 20 7 23 22 16) = 176")
            print("4. (AVG 176 20 1 16 7 6 22) = (176+20+1+16+7+6+22)/7 = 248/7 = 35")
            print("5. (MIN 35 23 14 24 7 16 24) = 7")
            print("6. (MAX 7 5 28 1 11) = 28")
            print("7. (SM 28 19 6 18 2 2 5 9) = (28+19+6+18+2+2+5+9) % 10 = 89 % 10 = 9")
            print("8. (MED 9 22 2 19 3) = median of [2,3,9,19,22] = 9")
        
        # Compute the correct answer with our regex-based parser
        result = eval_listops_expression(expr)
        print(f"\nComputed Result: {result}")
        
        if result == ground_truth_values.get(sample_id):
            print("✓ Ground truth is CORRECT")
        elif result == suggested_answers.get(sample_id):
            print("! Suggested answer is CORRECT")
        else:
            print("✗ Both ground truth and suggested answer are INCORRECT")
            
    except Exception as e:
        print(f"Error in evaluation: {e}")

def main():
    print("=== GROUND TRUTH VERIFICATION ===")
    
    for sample_id, expr in suspicious_samples.items():
        evaluate_and_print_steps(expr, sample_id)
        print("-" * 50)
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main() 