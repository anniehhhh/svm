import re
def validate_mystuff(whatisit):
# define our regex pattern for validation
    pattern = r"^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$" # you need to find out this pattern!!!
# We use the re.match function to test the password against the pattern
    match = re.match(pattern, whatisit)
# return True if the password matches the pattern, False otherwise
    return bool(match)
value = input("Enter the value")
print(validate_mystuff(value))