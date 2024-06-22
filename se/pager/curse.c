#include <ncurses.h>

// compile this file with the following command:
// gcc -o curse curse.c -lncurses

int main()
{	
    initscr();
    printw("Hello world\n");
    printw("Press any key to quit.");
    refresh();
    getch();
    endwin();
    return 0;
}
