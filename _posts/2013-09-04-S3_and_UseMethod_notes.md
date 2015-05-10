Classes/S3
==========

In these notes, we explain by example how the Class and UseMethod are
used, and how to overwrite function. These notes are taken from
<http://www.math.ncu.edu.tw/~chenwc/R_note/index.php?item=class>

Class and UseMethod
-------------------

This is a silly example, but it gives a hint to make a class by class()
and use generic functions by using UseMethod(). In R, you can define
your generic functions for print().

If a1 is a variable, then typing a1 (on the console) is the same as
print(a1). We will start definining some variable, a1, a2, and
illustrate the usage **class** and the **UseMethod**

    a1 <- 3.1415926
    class(a1) <- "myC1"
    a2 <- 6.1415926
    class(a2) <- "myC2"

    ## defining a generic function called my_fcn
    my_fcn <- function(x){
      UseMethod("my_usefcn", x)
    }

    ## defining the implmentation of my_fcn for class myC1 and myC2
    my_usefcn.myC1 <- function(x){
      x + 1
    }
    my_usefcn.myC2 <- function(x){
      x + 2
    }

**At First** when we call

    my_fcn(a1)

    ## [1] 4.141593
    ## attr(,"class")
    ## [1] "myC1"

    my_fcn(a2)

    ## [1] 8.141593
    ## attr(,"class")
    ## [1] "myC2"

**my\_fcn()** return the object's **attributes** since there is no
default *print()* function for classes *myC1* and *myC2*.

Similarly, when we type a1 and a2 in the console

    a1

    ## [1] 3.141593
    ## attr(,"class")
    ## [1] "myC1"

    a2

    ## [1] 6.141593
    ## attr(,"class")
    ## [1] "myC2"

we get the **attributes** since there is no default *print()* function
for classes *myC1* and *myC2*

**But now** let's define some *print()* function for these classes

    #Defining a print function for class myC1 --- Note the function names
    print.myC1 <- function(x, digits = 3){
      print(unclass(x), digits = digits)
    }
    #Defining a print function for class myC2
    print.myC2 <- function(x, digits = 6){
      print(unclass(x), digits = digits)
    }

and now when we call

    my_fcn(a1)

    ## [1] 4.14

    my_fcn(a2)

    ## [1] 8.14159

    print(a1)

    ## [1] 3.14

    print(a2)

    ## [1] 6.14159

    a1

    ## [1] 3.14

    a2

    ## [1] 6.14159

we get the nice R behavior and our newly defined objects of class *myC1*
and *myC2* print properly.

**Notes**: \* The attributes are discarded in *print()* by using
*unclass()* function \* the syntax for the function **name** of print
functions. There is period followed by the name of the class for which
the generic function is defined for. \* A more formal way of stating the
previous note, [is](http://www.pitt.edu/~njc23/Lecture4.pdf): To create
an S3 method write a function with the name generic.class, where generic
is a generic function name and class is the corresponding class for the
method. Examples of generic functions are *summary()*, *print()* and
*plot()*.

Summary and Print
-----------------

In, we define some gerenic functions for summary(), and print(), so they
can summary the results by the input's attribute and print it by the
summary's attribute, **NOT** the input's attribute. Note that class is
an attribute!

    summary.myC3 <- function(x){
      x <- x + 10
      class(x) <-"summary_C3" 
      x
    }
    summary.myC4 <- function(x){
      x <- x + 20
      class(x) <-"summary_C4" 
      x
    }
    print.summary_C3 <- function(x, digits = 3){
      cat("Result: ", format(x, digits = digits), "\n")
    }
    print.summary_C4 <- function(x, digits = 6){
      cat("Result: ", format(x, digits = digits), "\n")
    }

In the above definition, we have created a *summary* function for
classes *myC3* and *myC4*. In those function we have set the classes of
the object to *summary\_C3* and *summary\_C4* respectively. We then
defined print function for classes *summary\_C3* and *summary\_C4* which
will be used to print object of original classes *myC3* and *myC4*.
Although convoluted, this illustrates how we can "chain" classes
together to use function already defined for other classes. In the
following we provide examples to show how these functions are used and
the value they return.

    a1 <- 3.1415926
    class(a1) <- "myC3"
    a2 <- 6.1415926
    class(a2) <- "myC4"

Now we when we call the apply the summary function on these newly
defined objects, we get

    summary(a1)

    ## Result:  13.1

    summary(a2)

    ## Result:  26.1416

Overwrite Operators
-------------------

This is also the other silly example, but I want to demonstrate
overwrite functions in R and user defined function for binary operators.
In R, you can redefine an operator or use **%any%** to make a new one.
we define a unior operator of two sets in the following.

    a1 <- c("a", "b")
    a2 <- 4:7
    class(a1) <- "myC5"

    ### Overwrite "+" for a new class.
    "+.myC5" <- function(a, b) c(a, b) 

Now we apply the overloaded operator to our new defined object we get

    a1 + a2

    ## [1] "a" "b" "4" "5" "6" "7"

Similarly, we can overload the union operator as following

    ### Define a new one.
    "%union%" <- function(a, b) c(a, b)

and then use it as such

    a1 %union% a2

    ## [1] "a" "b" "4" "5" "6" "7"
