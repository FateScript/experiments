
## Pager

An implementation of simple pager.

### Files

-rw-r--r--@ 1 wangfeng  staff   3.7K Jun 27 19:26 console.py
-rw-r--r--@ 1 wangfeng  staff   260B Jun 21 20:28 curse.c
-rw-r--r--@ 1 wangfeng  staff   1.3K Jun 27 19:36 curse_pager.py
-rw-r--r--@ 1 wangfeng  staff   5.5K Jun 27 19:36 pager.py

`console.py` contains the implementation of console utils. For example, `get_press_key`/`terminal_size`.  
`pager.py` contains the implementation of the pager. It has two types of implementations: `console` and `curse`.  
`curse_pager.py` contains the implementation of the pager using the `curses` library.  
`curse.c` is a simple "Hello world" C program example using ncurses (python internal).  

### Usage

#### Start simple pager
```shell
python3 pager.py
```

#### Start pager with `curses`
```shell
python3 curse_pager.py pager.py
```

#### Compile and run `curse.c`
```shell
gcc -o curse curse.c -lncurses
./curse
```