class Person:
    def __init__(self, name, students):
        self.name = name
        self.students = students
    def __str__(self):
        return f"{self.name}({self.students})"
p1 = Person("CSEN3172", 40)
print(p1)