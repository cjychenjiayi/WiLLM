def text_to_dict(text):
    entries = text.split(';')
    result = {}
    for entry in entries:
        if ':' in entry:
            key, value = entry.split(':', 1)
            result[int(key.strip())] = value.strip()
    return result
dicts = [
    "1: Push&Pull; 2: Sweep; 3: Clap; 4:Slide; 5: Draw-Zigzag(Vertical); 6:Draw-N(Vertical);",
    "1: Draw-1; 2: Draw-2; 3: Draw-3; 4:Draw-4; 5: Draw-5; 6: Draw-6; 7:Draw-7; 8: Draw-8; 9: Draw-9; 0:Draw-0;",
    "1: Push&Pull; 2: Sweep; 3: Clap; 4:Draw-O(Vertical); 5: Draw-Zigzag(Vertical); 6: Draw-N(Vertical);",
    "1: Slide; 2: Draw-O(Horizontal); 3:Draw-Zigzag(Horizontal); 4: Draw-N(Horizontal); 5: Draw-Triangle(Horizontal); 6: Draw-Rectangle(Horizontal);",
    "1: Push&Pull; 2: Sweep; 3: Clap; 4:Draw-O(Horizontal); 5: Draw-Zigzag(Horizontal); 6: Draw-N(Horizontal);",
    "1: Push&Pull; 2: Sweep; 3: Clap; 4:Slide; 5: Draw-O(Horizontal); 6:Draw-Zigzag(Horizontal); 7: Draw-N(Horizontal); 8: Draw-Triangle(Horizontal); 9: Draw-Rectangle(Horizontal);",
    "1: Draw-O(Horizontal); 2:Draw-Zigzag(Horizontal); 3: Draw-N(Horizontal); 4: Draw-Triangle(Horizontal); 5: Draw-Rectangle(Horizontal);",
    "1: Push&Pull; 2: Sweep; 3:Clap; 4: Slide;",
    "1: Push&Pull"
]
folder_dict = {
    "20181109": 0,
    "20181112": 1,
    "20181115": 2,
    "20181116": 1,
    "20181117": 2,
    "20181118": 2,
    "20181121": 3,
    "20181127": 3,
    "20181128": 4,
    "20181130_user5_10_11": 5,
    "20181130_user12_13_14": 5,
    "20181130_user15_16_17": 5,
    "20181204": 5,
    "20181211": 4,
    "20181208": 7
}
folder_user_dict  ={
    "20181205": {
        2:6,
        3:3
        
    },
    "20181209": {
        2:8,
        6:5
    }
}
for key, value in folder_dict.items():
    folder_dict[key] = text_to_dict(dicts[value])
for key, value in folder_user_dict.items():
    for k, v in value.items():
        value[k] = text_to_dict(dicts[v])
    folder_user_dict[key] = value
# folder_dict = {
#     "20181109":"1: Push&Pull; 2: Sweep; 3: Clap; 4:Slide; 5: Draw-Zigzag(Vertical); 6:Draw-N(Vertical);",
#     "20181112":"1: Draw-1; 2: Draw-2; 3: Draw-3; 4:Draw-4; 5: Draw-5; 6: Draw-6; 7:Draw-7; 8: Draw-8; 9: Draw-9; 0:Draw-0;",
#     "20181115":"1: Push&Pull; 2: Sweep; 3: Clap; 4:Draw-O(Vertical); 5: Draw-Zigzag(Vertical); 6: Draw-N(Vertical);",
#     "20181116":"1: Draw-1; 2: Draw-2; 3: Draw-3; 4:Draw-4; 5: Draw-5; 6: Draw-6; 7:Draw-7; 8: Draw-8; 9: Draw-9; 0:Draw-0;",
#     "20181117":"1: Push&Pull; 2: Sweep; 3: Clap; 4:Draw-O(Vertical); 5: Draw-Zigzag(Vertical); 6: Draw-N(Vertical);",
#     "20181118":"1: Push&Pull; 2: Sweep; 3: Clap; 4:Draw-O(Vertical); 5: Draw-Zigzag(Vertical); 6: Draw-N(Vertical);",
#     "20181121":"1: Slide; 2: Draw-O(Horizontal); 3:Draw-Zigzag(Horizontal); 4: Draw-N(Horizontal); 5: Draw-Triangle(Horizontal); 6: Draw-Rectangle(Horizontal);",
#     "20181127":"1: Slide; 2: Draw-O(Horizontal); 3:Draw-Zigzag(Horizontal); 4: Draw-N(Horizontal); 5: Draw-Triangle(Horizontal); 6: Draw-Rectangle(Horizontal);",

# }
