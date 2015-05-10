Notes and Examples of Closures and Inheritance
=======================================================
Function have their own enviroments, and according to the this [link](http://digitheadslabnotebook.blogspot.ca/2011/06/environments-in-r.html) a Closure is a function that is packaged with some state. The previous link is a good post for understanding Enviroments in R, but that is not the focus of these notes.

Another point to keep in mind before looking into Closures, is that according to [R Language Definition](http://cran.r-project.org/doc/manuals/R-lang.html#Function-objects), Functions (or more precisely, function closures) have three basic components: a formal argument list, a body and an environment.

So before getting lost into the details, in R, functions, enviroments and closures are all related concept, we hope in these notes, we illustrate how to create, use closure instead of giving an excat defintions.

Creating a closure according to [Jeffrey A. Ryan](http://www.lemnica.com/esotericR/Introducing-Closures/)
------------------------------------------------
To create closures, we use the **environment** object in R. This allows for data and methods to reside _within_ the object instances, making self-aware behavior and selective inheritence easy.

### _Creating a stack in R_

* A stack implementation consists of three main components:
  1. a container variable --- a.k.a. the stack (.Data)
  2. a push method to add elements (push)
  3. a pop method to remove elements (pop)
  
We will provide the full R implementation, in what what follows first, and then explain code. For step-by-step explanation, see original post.


```r
# 1. Creating a new enviroment
stack <- new.env()
# 2. Creating a Data container (.Data) which is a vector which will contains our data, and which lives in the enviroment stack
stack$.Data <- vector()
# 3. Creating push method that acts on .Data. Note we use the << assignement operator
stack$push <- function(x) .Data <<- c(.Data, x)
# 4. Creating a pop method that acts on .Data
stack$pop <- function() {
    tmp <- .Data[length(.Data)]
    .Data <<- .Data[-length(.Data)]
    return(tmp)
}
# 5. Setting the enviroment of the push method to be the same as the enviroment of the stack object
environment(stack$push) <- as.environment(stack)
# 6. Setting the enviroment of the pop method to the same as the enviroment of the stack object
environment(stack$pop) <- as.environment(stack)
# 7. Setting the class attribute of our stack object to be 'stack' - this is not neccessarly step, but it will be needed later
class(stack) <- "stack"
```


The comments in the above code gives an step-by-step description on how to create a closure. However it's worth nothing that
 * If Step 5 and 6, were not included, then when we would have gotten an error when after step 4, we issue the following command.


```r
stack$push(1)
```

the error that we would have gotten is _object '.Data' not found_, and that is because we haven't matched the environment of the function to the object's environment. R isn't starting its search for .Data in the correct location. That is why step 5 and 6 and needed and we used the function _environment_ and _as.environment_.

Next, we are going to use S3 classes to create push and pop methods to make the calls look more like normal R. That is why step 7 above was needed. See Notes on S3 classes for further explanations


```r
push <- function(x, value, ...) UseMethod("push")
pop <- function(x, ...) UseMethod("pop")
push.stack <- function(x, value, ...) x$push(value)
pop.stack <- function(x) x$pop()
```


Finally, in order to make the creation of the above code easier, we will wrap it in a function, so that we can create a closure with one 1 line instead of going through these 7 steps every time. The final code will look as follow


```r
# This new_stack() becomes the constructor of the object whose class is 'stack'. Everything else in the code is the same as before
new_stack <- function() {
    stack <- new.env()
    stack$.Data <- vector()
    stack$push <- function(x) .Data <<- c(.Data, x)
    stack$pop <- function() {
        tmp <- .Data[length(.Data)]
        .Data <<- .Data[-length(.Data)]
        return(tmp)
    }
    environment(stack$push) <- as.environment(stack)
    environment(stack$pop) <- as.environment(stack)
    class(stack) <- "stack"
    stack
}
```


### A Simpler Example.

The following examples was taken from taken [here](from http://www.mail-archive.com/r-help@r-project.org/msg141704.html) and uses functions to create a closure. This example does not uses enviroment since the data lives inside the function and not at the same level as the functions. 


```r
account <- function(balance = 0) {
    function(d = 0, w = 0) {
        newbal <- balance + d - w
        balance <<- newbal
        newbal
    }
}
```


In the above example, *d* stands for deposit, and *w* stands for withdrawl. This is a function of function, that it, when we assigned it to a variable, that variable is a function as we will see next. In the following instanciation, we will create an account for **John** and the initial balance of this account will be 100.


```r
John <- account(100)
```

As we can see from the next statement, John is a function

```r
John
```

```
## function(d = 0, w = 0) {
##         newbal <- balance + d - w
##         balance <<- newbal
##         newbal
##     }
## <environment: 0x104347158>
```

and the default function call will give us the balance in the account as illustrated next

```r
John()
```

```
## [1] 100
```


Next, if John deposits 100, and makes a withdrawl for 50

```r
John(d = 100, w = 50)
```

John will be left with.....???

```r
John()
```

```
## [1] 150
```


In the next instanciation, we will create an account for Leo with 1000, and makes some action similar to John's account like above.

```r
## We create an account for Leo with 1000 in it
Leo <- account(1000)
# Viewing Leo account function
Leo
```

```
## function(d = 0, w = 0) {
##         newbal <- balance + d - w
##         balance <<- newbal
##         newbal
##     }
## <environment: 0x1049676d8>
```

Note that the function definition of Leo, although identical to to John, has a different *memory address*

```r
# we inspect the initial amount of Leo's account, and performs some
# transaction such as above for John
Leo()
```

```
## [1] 1000
```

```r
Leo(d = 1000, w = 50)
```

```
## [1] 1950
```

```r
Leo()
```

```
## [1] 1950
```

```r
Leo(d = 100, w = 500)
```

```
## [1] 1550
```

```r
Leo()
```

```
## [1] 1550
```


Inheritance - Continuation of the stack example
------------------------------------------------
Now, we are going extend the "stack" class with new functionality via inheritance.
Using the new_stack constructor, We are going to add "shift" and "unshift" methods and we can extend the "stack" object to a new class called "betterstack".


```r
# The new_betterstack() is constructor of the object whose class is 'betterstack' and that is child/inherited from the object with class 'stack'
new_betterstack <- function() {
    
    # creating a stack object
    stack <- new_stack()
    # setting the enviroment of the stack object to a variable stack_env
    stack_env <- as.environment(stack)
    # Defining the new shift methods to stack object
    stack$shift <- function(x) .Data <<- c(x, .Data)
    # Defining the new unshift methods to stack object
    stack$unshift <- function() {
        tmp <- .Data[1]
        .Data <<- .Data[-1]
        return(tmp)
    }
    # setting the enviroment of stack$shift to the enviroment of stack
    environment(stack$shift) <- stack_env
    # setting the enviroment of stack$unshift to the enviroment of stack
    environment(stack$unshift) <- stack_env
    # setting the class attribute of newly created stack object
    class(stack) <- c("betterstack", "stack")
    # returning the object stack
    stack
}
```


Next we create shift and unshift methods to make the calls look more like normal R, to be used with the objects whose class is "_betterstack_"


```r
shift <- function(x, value, ...) UseMethod("shift")
unshift <- function(x, ...) UseMethod("unshift")
shift.betterstack <- function(x, value, ...) x$shift(value)
unshift.betterstack <- function(x) x$unshift()
```


Now we illustrate the usage of the newly created "_betterstack_" object


```r
nb <- new_betterstack()
push(nb, 1:3)
nb$.Data
```

```
## [1] 1 2 3
```

```r
pop(nb)  # from the back
```

```
## [1] 3
```

```r
unshift(nb)  # from the front
```

```
## [1] 1
```

```r
shift(nb, 3)
push(nb, 1)
nb$.Data
```

```
## [1] 3 2 1
```



Finally, as Ryan mentioned in his post: Examples of Closures implementations can be found in the IBrokers package that interfaces the Interactive Brokers trading platform. See the twsConnect and eWrapper objects in the package on CRAN.

#### Further Notes on Knitr and Chunk's Options


* http://www.rstudio.com/ide/docs/authoring/using_markdown
* http://rpubs.com/gallery/options
* http://yihui.name/knitr/options
