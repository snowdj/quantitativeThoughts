R and Callback functions + Some HTML/CSS formatting
========================================================

A general explanation of what a callback function is can be found on [stackoverflow](http://stackoverflow.com/questions/11936451/what-is-a-callback-mechanism-and-how-it-applies-in-r). In these notes, we will try to explain how to create them, and use them in R. These notes are the summaries of [Duncan Temple Lang's note](http://developer.r-project.org/TaskHandlers.pdf). It leaves out all the stuff which i didn't understand, and some of the terminology might be wrong. simply due to my ignorance. This also focus on R-callback functions created and invoked from R.
__________________________________________________________

### Intro

The basic idea here is that we want to allow R functions to be automatically invoked at the end of each top-level expression. This might be usefull if one, for example, wants to have a call to _save.image()_ periodically invoked to reduce the potential for data loss if a crash were to occur subsequently. Rather than being based on time and the current semantics of the event loop, it makes more sense to call _save.image()_ at the end of a top-level task.


There are two basic functions with which one can add and remove R functions which are to be used as task callbacks that are to be invoked at the end of each top-level expression evaluation. These are __addTaskCallback()__ for registering a *handler*, and __removeTaskCallback()__ for removing one. Each function that is used as a task callback must accept at least four arguments.

#### Example of defining a callback function

```r
counter <- function() {
    ctr <- 0
    function(expr, value, ok, visible) {
        ctr <<- ctr + 1
        cat("Count", ctr, "\n")
        TRUE
    }
}
```


The above example, is an example of a __Closure__. Closures provide a convenient way to create a function that has access to any additional data that it needs. That
example, keeps tranck of a count of the number of times the callback has been invoked.

we can registers that R function to be called each time a top-level task is completed, by using _addTaskCallback()_, as follows:

```r
addTaskCallback(counter())
```


Sometime, it is more convenient to not use closures but to have R call the function with an additional argument that gives additional context. In these cases, we can supply a value for the data argument of _addTaskCallback()_ function. This object is stored by the callback mechanism and is given as a ﬁfth argument to the callback when it is invoked. We will see <a href="#LaterExample"> an exmple </a> of that later.

### Return value of a Callback

The return value has special signiﬁcance. It must be a **single logical value**.

* If the function returns __TRUE()__, the callback is maintained in the list of callbacks and it __will be__ invoked after the next task is completed. 
* If the function returns __FALSE()__, then the callback is removed from the list and __won’t be__ called again by this mechanism.

### Registering Multiple Handlers - the proper way!

The function __taskCallbackManager()__ is a function that can be used to create a manager which handles other callbacks. The functions allow one to _add()_ and _remove()_ functions to and from a list of callbacks. When a top-level task is completed, the managers central callback (_evaluate()_) is called by the C-level mechanism and this, in turn, evaluates each of the callbacks it manages. In the next section, we will take a look at some examples, which will hopefully make things a bit clearer and easier.


Examples
----------

The following is a simple example of how things work. The example handlers are not particularly interesting. But they illustrate how one can do different types of computations.

#### Example 2:

We continue our example by deﬁning a _times()_ function which will print a simple string to identify itself each time it is called.

```r
times <- function(total = 3, name = "a") {
    ctr <- 1
    function(expr, val, ok, visible) {
        cat("[Task ", name, "] ", ctr, "\n", sep = "")
        ctr <<- ctr + 1
        return(ctr <= total)
    }
}
```


Note that this function is that it removes itself from the handler list after it has been called a particular number of time, because the condition in the return will evalues to FALSE when the _ctr_ is larger than _total_

#### Example 3:

<a name="LaterExample"></a> A third example is periodic(), that will be called after each top-level task, but only does something every period calls. The function in this example takes an additional argument - _cmd_ - that is the value we give when registering the callback. In this case, we will pass an expression and periodic() will evaluate it.


```r
periodic <- function(period = 4, ctr = 0) {
# period - the number of calls between performing the action.
# ctr - can be specified to start at a different point.
    function(expr, value, ok, visible, cmd) {
        ctr <<- (ctr + 1) %% period # %% is the remainder operator
        if(ctr == 0) eval(cmd)
        return(TRUE)
        }
    }
```


### Usage Examples 

**NOTE: Results of the following examples, were [NOT produced naturally using knitr](https://groups.google.com/forum/#!topic/knitr/rrcJfcm0sbI). Instead the output was produced by cutting and pasting the code into from a regular R-session. The material that was copied back into this document is displayed in <span style="color:#4C886B">green</span>.**  

Now that we have defined _times_ and _periodic_ we will show how they can be used, using _addTaskCallback()_ or the more complex way using _taskCallbackManager()_; and what they produce when invoked in R
__________________________________________________

We ﬁrst start by adding a <u>collection</u> of _times()_ handlers. We given them different expiration numbers: 3, 4, 5, and 6. Also, we identify them as a, b, c, d. For the purpose of illustration, we ensure that none-are activated until we have registered them all. We do this by initially _suspending_ the manager and then adding the tasks. Then we _activate_ it again.


```r
# example of using taskCallbackManager() to register a collection of handlers
h <- taskCallbackManager()
h$suspend()
h$add(times())
```

```
## [1] "1"
```

```r
h$add(times(4,"b"))
```

```
## [1] "2"
```

```r
h$add(times(5,"c"))
```

```
## [1] "3"
```

```r
h$add(times(6,"d"))
```

```
## [1] "4"
```

```r
h$suspend(FALSE)
## [Task a] 1
## [Task b] 1
## [Task c] 1
## [Task d] 1
```



The output below the suspend(FALSE) is from each of the handlers giving their counts and identiﬁers.


Next, we add a _periodic()_ handler. We specify user data that R will pass to this function when it calls it. This is an expression that the handler function (i.e. the function returned by calling periodic()) will evaluate every _4^(th)_ call.


```r
addTaskCallback(periodic(), quote(print("ok")), name = "book3")
```

```
## book3 
##     2
```

```r
## [Task a] 2
## [Task b] 2
## [Task c] 2
## [Task d] 2
```


Again, the output below the result is from the handlers. The most recently added handler in this call (the function returned from calling _periodic()_) does not generate any output. Note that after defining the function _periodic()_, the _ctr_ has value of 1. So we must wait _<u>another</u> 3 calls_ for it to perform its real action.


We can continue to give regular R commands and see how the handlers work. We issue arbitrary commands and look at the output from the handlers.

```r
sum(rnorm(10))
```

```
## [1] -5.718
```

```r
## [Task a] 3
## [Task b] 3
## [Task c] 3
## [Task d] 3
```

At this point, the ﬁrst timer (a) - 1st callback/handler-  has expired having reached its maximal count of 3. It has been removed from the list and so will not appear in any subsequent output. we continue with issuing R-command


```r
sqrt(9)
```

```
## [1] 3
```

```r
## [Task b] 4
## [Task c] 4
## [Task d] 4
```


At this point, handler ‘b’ - 2nd callback/handler- has also expired and is removed.

```r
length(objects())
```

```
## [1] 3
```

```r
## [Task c] 5
## [Task d] 5
## [1] "ok"
```


After the last command, Handler ‘c’ has expired. Also, since this is the 4-th call, the _periodic()_ handler kicks in and evaluates its _cmd_ argument. This is the expression print("ok") and gives rise to the last line of the output. Now the _ctr_ variable in the _periodic()_ function has been reset to 0, and so we must wait 4 calls for it to perform its action again.

Finally, when we keep on typing the following 4 commands

```r
gamma(4)
```

```
## [1] 6
```

```r
## [Task d] 6
```


```r
gamma(4)
```

```
## [1] 6
```

```r
gamma(4)
```

```
## [1] 6
```

```r
gamma(4)
```

```
## [1] 6
```

```r
## [1] "ok"
```


After the ﬁrst of these calls, handler ‘d’ expires. The periodic() handler is still active. After the 4-th of these calls, it generates more output. And this will continue ad inﬁnitum.

### Removing Handlers

Removing handlers is quite simple. We can use either position indices or names. Names are preferred since positions change when other callbacks are removed.


```r
removeTaskCallback("book3")
```

```
## [1] TRUE
```


### Notes that were usefull when writting this document

1. To underlying a text, we wrapped the text with the underline HTML tag, as follows:  < u > text < /u > (without the spaces: right after the < , and right before the > brackets)

2. To link a part of this document to another, we added anchar tags as described in [this document](http://help.typepad.com/anchor-tags.html). The following is an example:
< a href="#myReference" > wordToBeClicked < /a >; and then later on where we want the link to directed to, we add < a name="myReference">< /a >. (without the spaces: right after the < , and right before the > brackets)

3. To Color a specific part of a _text_, we wrapped the text as follow: < span style="color:#4C886B" >text< /span > (without the spaces: right after the < , and right before the > brackets)

4. To convert from rgb color to Hexidecimal color code, I have used the converter on [this site](http://easycalculation.com/rgb-coder.php). Color:#4C886B is the <span style="color:#4C886B"> green </span> color used for the comments.

Thanks to Yinui Xie, creator of knitr package, who advised on the first 2 notes above, and on knitr limitation. He also brought to my attention that the giant behind markdown is HTML, and so I can include HTML tags directly in any RMarkdown and Knitr will take care of the rest.
