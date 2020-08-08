#!/usr/bin/env python
# coding: utf-8

# In[1]:


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
print(quicksort([3,6,8,10,1,2,1]))


# In[2]:


x = 3
print(x, type(x))


# In[3]:


print(x + 1)
print(x - 1)
print(x * 2)
print(x ** 2)


# In[4]:


x += 1
print(x)
x *= 2
print(x)


# In[5]:


y = 2.5
print(type(y))
print(y, y + 1, y * 2, y ** 2)


# In[6]:


t, f = True, False
print(type(t))


# In[7]:


print(t and f)
print(t or f) 
print(not t)
print(t != f)


# In[8]:


hello = 'hello'
world = "world"
print(hello, len(hello))


# In[9]:


hw = hello + ' ' + world 
print(hw)


# In[10]:


hw12 = '{} {} {}'.format(hello, world, 12)
print(hw12)


# In[11]:


s = "hello"
print(s.capitalize())
print(s.upper()) 


# In[12]:


print(s.rjust(7)) 
print(s.center(7)) 
print(s.replace('l', '(ell)')) 
print('  world '.strip())


# In[13]:


xs = [3, 1, 2]
print(xs, xs[2])
print(xs[-1]) 


# In[14]:


xs[2] = 'foo'
print(xs)


# In[15]:


xs.append('bar')
print(xs) 


# In[16]:


x = xs.pop() 
print(x, xs)


# In[19]:


nums = list(range(5)) 
print(nums)
print(nums[2:4])
print(nums[2:])
print(nums[:2])
print(nums[:])
print(nums[:-1])
nums[2:4] = [8, 9]
print(nums)
      


# In[20]:


animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)


# In[22]:


animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))


# In[24]:


nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
    print(squares)


# In[26]:


nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)


# In[27]:


nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2for x in nums if x % 2 == 0]
print(even_squares)


# In[28]:


d = {'cat': 'cute', 'dog': 'furry'}
print(d['cat']) 
print('cat'in d) 


# In[29]:


d['fish'] = 'wet'
print(d['fish'])


# In[30]:


print(d['monkey'])


# In[31]:


print(d.get('monkey', 'N/A'))
print(d.get('fish', 'N/A'))


# In[32]:


del d['fish'] 
print(d.get('fish', 'N/A'))


# In[33]:


d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A {} has {} legs'.format(animal, legs))


# In[34]:


nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2for x in nums if x % 2 == 0}
print(even_num_to_square)


# In[35]:


animals = {'cat', 'dog'}
print('cat'in animals)
print('fish'in animals) 


# In[36]:


animals.add('fish')      
print('fish'in animals)
print(len(animals))


# In[37]:


animals.add('cat')      
print(len(animals)) 
animals.remove('cat')
print(len(animals))


# In[39]:


animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))


# In[41]:


from math import sqrt
print({int(sqrt(x)) 
       for x in range(30)})


# In[43]:


d = {(x, x + 1): x for x in range(10)}  
t = (5, 6) 
print(type(t))
print(d[t])
print(d[(1, 2)])


# In[48]:


def sign(x):
    if x > 0:
        return'positive'
    elif x < 0:
        return'negative'
    else:
        return'zero'
    
for x in [-1, 0, 1]:print(sign(x))


# In[56]:


def hello(name, loud=False):
    if loud:print('HELLO, {}'.format(name.upper()))
    else: print('Hello, {}!'.format(name))
hello('Bob')
hello('Fred', loud=True)


# In[69]:


class Greeter:
    def __init__(self, name):
        self.name = name
    def greet(self, loud=False):
        if loud:
            print('HELLO, {}'.format(self.name.upper()))
        else:
            print('Hello, {}!'.format(self.name))
g = Greeter('Fred')
g.greet()
g.greet(loud=True)
        


# In[71]:


import numpy as np
a = np.array([1, 2, 3])
print(type(a), a.shape, a[0], a[1], a[2])
a[0] = 5
print(a)


# In[72]:


b = np.array([[1,2,3],[4,5,6]])   
print(b)


# In[73]:


print(b.shape)
print(b[0, 0], b[0, 1], b[1, 0])


# In[74]:


a = np.zeros((2,2))  
print(a)


# In[75]:


b = np.ones((1,2)) 
print(b)


# In[76]:


c = np.full((2,2), 7) 
print(c)


# In[77]:


d = np.eye(2)        
print(d)


# In[78]:


e = np.random.random((2,2))
print(e)


# In[79]:


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]
print(b)


# In[80]:


print(a[0, 1])
b[0, 0] = 77
print(a[0, 1]) 


# In[81]:


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)


# In[82]:


row_r1 = a[1, :] 
row_r2 = a[1:2, :] 
row_r3 = a[[1], :]
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)
print(row_r3, row_r3.shape)


# In[83]:


col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print()
print(col_r2, col_r2.shape)


# In[84]:


a = np.array([[1,2], [3, 4], [5, 6]])
print(a[[0, 1, 2], [0, 1, 0]])
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))


# In[86]:


print(a[[0, 0], [1, 1]])
print(np.array([a[0, 1], a[0, 1]]))


# In[88]:


a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)


# In[89]:


b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])


# In[90]:


a[np.arange(4), b] += 10
print(a)


# In[92]:


import numpy as np
a = np.array([[1,2], [3, 4], [5, 6]])
bool_idx = (a > 2)
print(bool_idx)


# In[93]:


print(a[bool_idx])
print(a[a > 2])


# In[94]:


x = np.array([1, 2])
y = np.array([1.0, 2.0])
z = np.array([1, 2], dtype=np.int64)
print(x.dtype, y.dtype, z.dtype)


# In[95]:


x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
print(x + y)
print(np.add(x, y))


# In[96]:


print(x - y)
print(np.subtract(x, y))


# In[97]:


print(x * y)
print(np.multiply(x, y))


# In[98]:


print(x / y)
print(np.divide(x, y))


# In[99]:


print(np.sqrt(x))


# In[104]:


x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
v = np.array([9,10])
w = np.array([11, 12])
print(v.dot(w))
print(np.dot(v, w))


# In[105]:


print(v @ w)


# In[106]:


print(x.dot(v))
print(np.dot(x, v))
print(x @ v)


# In[107]:


print(x.dot(y))
print(np.dot(x, y))
print(x @ y)


# In[108]:


x = np.array([[1,2],[3,4]])
print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1)) 


# In[109]:


print(x)
print("transpose\n", x.T)


# In[110]:


v = np.array([[1,2,3]])
print(v )
print("transpose\n", v.T)


# In[112]:


x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)
for i in range(4):
    y[i, :] = x[i, :] + v
    print(y)


# In[113]:


vv = np.tile(v, (4, 1))
print(vv)  


# In[114]:


y = x + vv
print(y)


# In[115]:


import numpy as np
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v
print(y)


# In[117]:


v = np.array([1,2,3]) 
w = np.array([4,5]) 
print(np.reshape(v, (3, 1)) * w)


# In[118]:


x = np.array([[1,2,3], [4,5,6]])
print(x + v)


# In[119]:


print((x.T + w).T)


# In[120]:


print(x + np.reshape(w, (2, 1)))


# In[121]:


print(x * 2)


# In[122]:


import matplotlib.pyplot as plt
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)
plt.plot(x, y)


# In[123]:


y_sin = np.sin(x)
y_cos = np.cos(x)
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])


# In[124]:


x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.title('Sine')
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')
plt.show()


# In[ ]:




