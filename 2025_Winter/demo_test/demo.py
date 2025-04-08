dir_name = "./demo_test/"
filename = "1.txt"
with open(filename, "r", encoding="utf-8") as f:
    content = f.read()

print(content)