from env.tasks import task_easy, task_medium, task_hard

if __name__ == "__main__":
    print("Running baseline...")

    print("Easy Score:", round(task_easy(), 2))
    print("Medium Score:", round(task_medium(), 2))
    print("Hard Score:", round(task_hard(), 2))
