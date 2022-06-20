#!/usr/bin/env python
import gkeepapi
import todotxtio

todo_tile = "/home/k/Desktop/todo.txt"

keep = gkeepapi.Keep()
success = keep.login('ismailezzaki96@gmail.com', 'mhkmsbxpvohndjqk')

gnotes = keep.find(labels=[keep.findLabel('Planning')])
todos = []
for note in gnotes:
    if (note.title == "My Tasks"):
        glistitems = note.items
        for item in glistitems:
            todos.append(todotxtio.Todo(text=item.text))
            print(item.text)



todotxtio.to_file(todo_tile, todos)
