import tensorflow as tf

# The tf.string data type
print(tf.constant(u"Thanks 😊"))

print(tf.constant([u"Your're", u"Welcome!"]).shape)

# Note: When using python to construct strings, the handling of unicode differs between v2 and v3.
# In v2, unicode strings are indicated by the "u" prefix, as above. In v3, strings are unicode-encoded by default.

# Representign Unicode
# Unicode string, represented as a UTF-8 encoded string scalar.
text_utf8 = tf.constant(u"语言处理")
print("text_utf8: {}".format(text_utf8))

# Unicode string, represented as a UTF-16-BE encoded string scalar.
text_utf16be = tf.constant(u"语言处理".encode("UTF-16-BE"))
print("text_utf16be: {}".format(text_utf16be))

# Unicode string, represented as a vector of Unicode code points.
text_chars = tf.constant([ord(char) for char in u"语言处理"])
print("text_chars: {}".format(text_chars))

# Converting between representations