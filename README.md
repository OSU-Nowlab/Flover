# Flover
A novel temporal fusion framework for propelling autoregressive model inference by leveraging the temporality property of inferring generative tasks, delivering superior and more fine-grained parallelism beyond all current solutions.
The code will be release at July.

**Structure**
![Example Image](images/Flover.png)
---

**Comparison with FT in constant request time interval**

![Example Image](images/compare_in_fix_interval.png)
---


**Comparison with FT in random request time interval**

![Example Image](images/compare_in_poisson.png)
---



**Comparison with FT in random request output length (sampled from a uniform distribution)**

![Example Image](images/compare_in_variable_length.png)
---



**Working with tensor parallelism**

![Example Image](images/compare_in_distributed.png)
---
