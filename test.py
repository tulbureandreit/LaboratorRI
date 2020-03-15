import os


if os.path.isdir(r"C:\Users\dell\PycharmProjects\untitled\venv\dataset\Good"):
    for i, filename in enumerate(os.listdir(r"C:\Users\dell\PycharmProjects\untitled\venv\dataset\Good")):
        os.rename(r"C:\Users\dell\PycharmProjects\untitled\venv\dataset\Good" + "/" + filename, r"C:\Users\dell\PycharmProjects\untitled\venv\dataset\Good" + "/" +str(i) + ".bmp")