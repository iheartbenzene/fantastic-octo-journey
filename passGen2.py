import random
import string


def randomString(stringLength=15):
    """Generate a random string of fixed length """
    letters = string.ascii_letters
    digits = string.digits
    characters = string.punctuation
    return ''.join(random.SystemRandom().choice(letters + digits + characters) for i in range(stringLength))
print ("Password choice 1 is ", randomString() )
print ("Password choice 2 is ", randomString(20) )
print ("Password choice 3 is ", randomString(25) )