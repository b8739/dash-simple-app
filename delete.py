import time


str = "A man, a plan, a canal: Panama"


def isPalindrome(str):
    start_time = time.process_time()

    idx = -1
    for c in str:
        if c.isalnum() == False:
            continue
        while str[idx].lower().isalnum() == False:
            idx -= 1

        if c.lower() != str[idx].lower():
          
            return False
        idx -= 1
    end_time = time.process_time()

    return True


isPalindrome(str)
