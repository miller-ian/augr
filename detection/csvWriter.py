import csv



try:
    with open("annotationsDROID.txt", "a+") as file:
        while True:
            path = input("Enter path to the image: " )
            left = input("Enter left-bound x val: ")
            right = input("Enter right-bound x val: ")
            bottom = input("Enter bottom-bound y val: ")
            top = input("Enter top-bound y val: ")
            className = input("Enter class name: ")
            file.write(path + "," + left + "," + right + "," + bottom + "," + top + "," + className + "\n")
except KeyboardInterrupt:
    print("CSV file-builder closing...")

