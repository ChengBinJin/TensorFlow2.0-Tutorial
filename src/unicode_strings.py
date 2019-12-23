import tensorflow as tf

# The tf.string data type
print(tf.constant(u"Thanks 😊"))

print(tf.constant([u"Your're", u"Welcome!"]).shape)

# Note: When using python to construct strings, the handling of unicode differs between v2 and v3.
# In v2, unicode strings are indicated by the "u" prefix, as above. In v3, strings are unicode-encoded by default.

# Representign Unicode
# Unicode string, represented as a UTF-8 encoded string scalar.
text_utf8 = tf.constant(u"语言处理")
print("\ntext_utf8: {}".format(text_utf8))

# Unicode string, represented as a UTF-16-BE encoded string scalar.
text_utf16be = tf.constant(u"语言处理".encode("UTF-16-BE"))
print("\ntext_utf16be: {}".format(text_utf16be))

# Unicode string, represented as a vector of Unicode code points.
text_chars = tf.constant([ord(char) for char in u"语言处理"])
print("\ntext_chars: {}".format(text_chars))

# Converting between representations
print("\ntf.strings.unicode_decode(text_utf8, input_encoding='UTF-8'): ")
print(tf.strings.unicode_decode(text_utf8, input_encoding='UTF-8'))

print("\ntf.strings.unicode_encode(text_chars, output_encoding='UTF-8')")
print(tf.strings.unicode_encode(text_chars, output_encoding='UTF-8'))

print("\nprint(tf.strings.unicode_transcode(text_utf8, input_encoding='UTF8', output_encoding='UTF-16-BE'))")
print(tf.strings.unicode_transcode(text_utf8,
                                   input_encoding='UTF8',
                                   output_encoding='UTF-16-BE'))

# Batch dimensions
# A batch of Unicode strings, each represented as a UTF8-encoded string.
batch_utf8 = [s.encode('UTF-8') for s in [u'hÃllo',  u'What is the weather tomorrow',  u'Göödnight', u'😊']]

print('batch_utf8: ')
for chars in batch_utf8:
    print(chars)

batch_chars_ragged = tf.strings.unicode_decode(batch_utf8, input_encoding='UTF-8')

for sentence_chars in batch_chars_ragged.to_list():
    print(sentence_chars)

batch_chars_padded = batch_chars_ragged.to_tensor(default_value=-1)
print(batch_chars_padded.numpy())

batch_chars_sparse = batch_chars_ragged.to_sparse()

print(tf.strings.unicode_encode([[99, 97, 116], [100, 111, 103], [99, 111, 119]], output_encoding='UTF-8'))
print(tf.strings.unicode_encode(batch_chars_ragged, output_encoding='UTF-8'))

print(tf.strings.unicode_encode(tf.RaggedTensor.from_sparse(batch_chars_sparse), output_encoding='UTF-8'))
print(tf.strings.unicode_encode(tf.RaggedTensor.from_tensor(batch_chars_padded, padding=-1), output_encoding='UTF-8'))

# Unicode operations
# Note that the final character takes up 4 bytes in UTF8.
thanks = u'Thanks 😊'.encode('UTF-8')
num_bytes = tf.strings.length(thanks).numpy()
num_chars = tf.strings.length(thanks, unit='UTF8_CHAR').numpy()
print('{} bytes; {} UTF-8 characters'.format(num_bytes, num_chars))

# Character substrings
# default: unit='BYTE'. with len=1, we return a single byte.
print(tf.strings.substr(thanks, pos=7, len=1).numpy())

