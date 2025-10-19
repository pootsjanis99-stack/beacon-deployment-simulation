## Simulation Controls
# Mouse
- Left Click:
- Select a drop (yellow/red circle) or orange anchor.
- If clicking on empty terrain, sets the last_click position (used for placing new drops, anchors, or cars).

# Keyboard
- D:
- If you clicked on a drop: advances that drop’s stage (1 → 2 → 3 → 4).
- If you clicked on empty terrain: creates a new yellow drop (stage 1) at that spot.
- If you haven’t clicked anywhere: advances all drops at once.
- O:
- Places an orange anchor node at the last clicked position.
- P:
- Deletes the currently selected orange anchor.
- R:
- Removes the currently selected red beacon node.
- C:
- Spawns a car (blue rectangle) at the last clicked position.
- U:
- Draws a straight blue line (route) from the nearest car to the selected orange anchor.
- Z:
- Toggles zoom mode on/off.
- When ON: view zooms to fit all active nodes, anchors, cars, and triangulation outlines.
- When OFF: view shows the full map.
- X:
- Resets the entire map (clears drops, anchors, cars, routes).

# Buttons (bottom of the window)
- Generate New Map (Reset)
- Clears everything and generates a fresh random terrain.
- Add Adjacent Grid
- Expands the map by adding another terrain grid to the right.
