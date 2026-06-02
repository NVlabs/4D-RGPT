def multipl_choice_accuracy(row) -> bool:
    return (row["output"][-1] in "ABCDE") and (row["output"][-1] == row["answer"])
