class libbook:
    def __init__(self):
        self.books=[]
        self.nobooks=0
    def addbooks(self,book):
        self.books.append(book)
        self.nobooks= len(self.books)
    def bookdetails(self):
        print(f"the lib has {self.nobooks}")  
        for book in self.books:
            print(book)


obj1 = libbook()
obj1.addbooks("harry potter")
obj1.addbooks("inferno")
obj1.addbooks("the da vinci code")

obj1.bookdetails()