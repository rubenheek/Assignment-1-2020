import sys
import process


if __name__ == "__main__":
    '''
    Entry point for the script
    '''
    if len(sys.argv) < 2:
        print("Please provide an argument to indicate which matcher should be used")
        exit(1)

    match_type = 0
    try:
        match_type = int(sys.argv[1])
    except ValueError as e:
        print("Match type provided is not a valid number")
        exit(1)
    
    if match_type in range(4):
        process.process(match_type)
    else:
        print("Please provide a match type between 0 and 3, including")
        exit(1) 
