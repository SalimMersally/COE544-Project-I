??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.22v2.9.1-132-g18960c44ad38??	
?
Adam/dense_136/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*&
shared_nameAdam/dense_136/bias/v
{
)Adam/dense_136/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_136/bias/v*
_output_shapes
:>*
dtype0
?
Adam/dense_136/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?>*(
shared_nameAdam/dense_136/kernel/v
?
+Adam/dense_136/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_136/kernel/v*
_output_shapes
:	?>*
dtype0
?
Adam/dense_135/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_135/bias/v
|
)Adam/dense_135/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_135/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_135/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*(
shared_nameAdam/dense_135/kernel/v
?
+Adam/dense_135/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_135/kernel/v*
_output_shapes
:	@?*
dtype0
?
Adam/dense_134/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_134/bias/v
{
)Adam/dense_134/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_134/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_134/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*(
shared_nameAdam/dense_134/kernel/v
?
+Adam/dense_134/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_134/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/conv2d_52/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_52/bias/v
|
)Adam/conv2d_52/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_52/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_52/kernel/v
?
+Adam/conv2d_52/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_51/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_51/bias/v
{
)Adam/conv2d_51/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_51/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_51/kernel/v
?
+Adam/conv2d_51/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_50/bias/v
{
)Adam/conv2d_50/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_50/kernel/v
?
+Adam/conv2d_50/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/dense_136/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*&
shared_nameAdam/dense_136/bias/m
{
)Adam/dense_136/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_136/bias/m*
_output_shapes
:>*
dtype0
?
Adam/dense_136/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?>*(
shared_nameAdam/dense_136/kernel/m
?
+Adam/dense_136/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_136/kernel/m*
_output_shapes
:	?>*
dtype0
?
Adam/dense_135/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_135/bias/m
|
)Adam/dense_135/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_135/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_135/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*(
shared_nameAdam/dense_135/kernel/m
?
+Adam/dense_135/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_135/kernel/m*
_output_shapes
:	@?*
dtype0
?
Adam/dense_134/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_134/bias/m
{
)Adam/dense_134/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_134/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_134/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*(
shared_nameAdam/dense_134/kernel/m
?
+Adam/dense_134/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_134/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/conv2d_52/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_52/bias/m
|
)Adam/conv2d_52/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_52/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_52/kernel/m
?
+Adam/conv2d_52/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_51/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_51/bias/m
{
)Adam/conv2d_51/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_51/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_51/kernel/m
?
+Adam/conv2d_51/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_50/bias/m
{
)Adam/conv2d_50/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_50/kernel/m
?
+Adam/conv2d_50/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/kernel/m*&
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
t
dense_136/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*
shared_namedense_136/bias
m
"dense_136/bias/Read/ReadVariableOpReadVariableOpdense_136/bias*
_output_shapes
:>*
dtype0
}
dense_136/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?>*!
shared_namedense_136/kernel
v
$dense_136/kernel/Read/ReadVariableOpReadVariableOpdense_136/kernel*
_output_shapes
:	?>*
dtype0
u
dense_135/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_135/bias
n
"dense_135/bias/Read/ReadVariableOpReadVariableOpdense_135/bias*
_output_shapes	
:?*
dtype0
}
dense_135/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*!
shared_namedense_135/kernel
v
$dense_135/kernel/Read/ReadVariableOpReadVariableOpdense_135/kernel*
_output_shapes
:	@?*
dtype0
t
dense_134/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_134/bias
m
"dense_134/bias/Read/ReadVariableOpReadVariableOpdense_134/bias*
_output_shapes
:@*
dtype0
}
dense_134/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*!
shared_namedense_134/kernel
v
$dense_134/kernel/Read/ReadVariableOpReadVariableOpdense_134/kernel*
_output_shapes
:	?@*
dtype0
u
conv2d_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_52/bias
n
"conv2d_52/bias/Read/ReadVariableOpReadVariableOpconv2d_52/bias*
_output_shapes	
:?*
dtype0
?
conv2d_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_52/kernel
~
$conv2d_52/kernel/Read/ReadVariableOpReadVariableOpconv2d_52/kernel*'
_output_shapes
:@?*
dtype0
t
conv2d_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_51/bias
m
"conv2d_51/bias/Read/ReadVariableOpReadVariableOpconv2d_51/bias*
_output_shapes
:@*
dtype0
?
conv2d_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_51/kernel
}
$conv2d_51/kernel/Read/ReadVariableOpReadVariableOpconv2d_51/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_50/bias
m
"conv2d_50/bias/Read/ReadVariableOpReadVariableOpconv2d_50/bias*
_output_shapes
: *
dtype0
?
conv2d_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_50/kernel
}
$conv2d_50/kernel/Read/ReadVariableOpReadVariableOpconv2d_50/kernel*&
_output_shapes
: *
dtype0

NoOpNoOp
?e
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?e
value?eB?e B?e
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
?
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias
 ,_jit_compiled_convolution_op*
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
?
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias
 ;_jit_compiled_convolution_op*
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses* 
?
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses* 
?
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias*
?
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias*
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator* 
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

ekernel
fbias*
Z
0
1
*2
+3
94
:5
N6
O7
V8
W9
e10
f11*
Z
0
1
*2
+3
94
:5
N6
O7
V8
W9
e10
f11*
* 
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ltrace_0
mtrace_1
ntrace_2
otrace_3* 
6
ptrace_0
qtrace_1
rtrace_2
strace_3* 
* 
?
titer

ubeta_1

vbeta_2
	wdecay
xlearning_ratem?m?*m?+m?9m?:m?Nm?Om?Vm?Wm?em?fm?v?v?*v?+v?9v?:v?Nv?Ov?Vv?Wv?ev?fv?*

yserving_default* 

0
1*

0
1*
* 
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_50/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_50/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

*0
+1*

*0
+1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_51/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_51/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

90
:1*

90
:1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_52/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_52/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

N0
O1*

N0
O1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEdense_134/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_134/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

V0
W1*

V0
W1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEdense_135/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_135/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

e0
f1*

e0
f1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEdense_136/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_136/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
0
1
2
3
4
5
6
7
	8

9
10*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
?}
VARIABLE_VALUEAdam/conv2d_50/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_50/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_51/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_51/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_52/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_52/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_134/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_134/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_135/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_135/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_136/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_136/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_50/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_50/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_51/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_51/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_52/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_52/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_134/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_134/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_135/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_135/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/dense_136/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_136/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_conv2d_50_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_50_inputconv2d_50/kernelconv2d_50/biasconv2d_51/kernelconv2d_51/biasconv2d_52/kernelconv2d_52/biasdense_134/kerneldense_134/biasdense_135/kerneldense_135/biasdense_136/kerneldense_136/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_638673
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_50/kernel/Read/ReadVariableOp"conv2d_50/bias/Read/ReadVariableOp$conv2d_51/kernel/Read/ReadVariableOp"conv2d_51/bias/Read/ReadVariableOp$conv2d_52/kernel/Read/ReadVariableOp"conv2d_52/bias/Read/ReadVariableOp$dense_134/kernel/Read/ReadVariableOp"dense_134/bias/Read/ReadVariableOp$dense_135/kernel/Read/ReadVariableOp"dense_135/bias/Read/ReadVariableOp$dense_136/kernel/Read/ReadVariableOp"dense_136/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_50/kernel/m/Read/ReadVariableOp)Adam/conv2d_50/bias/m/Read/ReadVariableOp+Adam/conv2d_51/kernel/m/Read/ReadVariableOp)Adam/conv2d_51/bias/m/Read/ReadVariableOp+Adam/conv2d_52/kernel/m/Read/ReadVariableOp)Adam/conv2d_52/bias/m/Read/ReadVariableOp+Adam/dense_134/kernel/m/Read/ReadVariableOp)Adam/dense_134/bias/m/Read/ReadVariableOp+Adam/dense_135/kernel/m/Read/ReadVariableOp)Adam/dense_135/bias/m/Read/ReadVariableOp+Adam/dense_136/kernel/m/Read/ReadVariableOp)Adam/dense_136/bias/m/Read/ReadVariableOp+Adam/conv2d_50/kernel/v/Read/ReadVariableOp)Adam/conv2d_50/bias/v/Read/ReadVariableOp+Adam/conv2d_51/kernel/v/Read/ReadVariableOp)Adam/conv2d_51/bias/v/Read/ReadVariableOp+Adam/conv2d_52/kernel/v/Read/ReadVariableOp)Adam/conv2d_52/bias/v/Read/ReadVariableOp+Adam/dense_134/kernel/v/Read/ReadVariableOp)Adam/dense_134/bias/v/Read/ReadVariableOp+Adam/dense_135/kernel/v/Read/ReadVariableOp)Adam/dense_135/bias/v/Read/ReadVariableOp+Adam/dense_136/kernel/v/Read/ReadVariableOp)Adam/dense_136/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_639188
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_50/kernelconv2d_50/biasconv2d_51/kernelconv2d_51/biasconv2d_52/kernelconv2d_52/biasdense_134/kerneldense_134/biasdense_135/kerneldense_135/biasdense_136/kerneldense_136/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_50/kernel/mAdam/conv2d_50/bias/mAdam/conv2d_51/kernel/mAdam/conv2d_51/bias/mAdam/conv2d_52/kernel/mAdam/conv2d_52/bias/mAdam/dense_134/kernel/mAdam/dense_134/bias/mAdam/dense_135/kernel/mAdam/dense_135/bias/mAdam/dense_136/kernel/mAdam/dense_136/bias/mAdam/conv2d_50/kernel/vAdam/conv2d_50/bias/vAdam/conv2d_51/kernel/vAdam/conv2d_51/bias/vAdam/conv2d_52/kernel/vAdam/conv2d_52/bias/vAdam/dense_134/kernel/vAdam/dense_134/bias/vAdam/dense_135/kernel/vAdam/dense_135/bias/vAdam/dense_136/kernel/vAdam/dense_136/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_639333??
?
?
E__inference_conv2d_50_layer_call_and_return_conditional_losses_638206

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_638998

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?/
?
I__inference_sequential_41_layer_call_and_return_conditional_losses_638597
conv2d_50_input*
conv2d_50_638561: 
conv2d_50_638563: *
conv2d_51_638567: @
conv2d_51_638569:@+
conv2d_52_638573:@?
conv2d_52_638575:	?#
dense_134_638580:	?@
dense_134_638582:@#
dense_135_638585:	@?
dense_135_638587:	?#
dense_136_638591:	?>
dense_136_638593:>
identity??!conv2d_50/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!dense_134/StatefulPartitionedCall?!dense_135/StatefulPartitionedCall?!dense_136/StatefulPartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCallconv2d_50_inputconv2d_50_638561conv2d_50_638563*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_638206?
 max_pooling2d_48/PartitionedCallPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_638161?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0conv2d_51_638567conv2d_51_638569*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_638224?
 max_pooling2d_49/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_638173?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_52_638573conv2d_52_638575*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_638242?
 max_pooling2d_50/PartitionedCallPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_638185?
flatten_17/PartitionedCallPartitionedCall)max_pooling2d_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_638255?
!dense_134/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_134_638580dense_134_638582*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_134_layer_call_and_return_conditional_losses_638268?
!dense_135/StatefulPartitionedCallStatefulPartitionedCall*dense_134/StatefulPartitionedCall:output:0dense_135_638585dense_135_638587*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_135_layer_call_and_return_conditional_losses_638285?
dropout_23/PartitionedCallPartitionedCall*dense_135/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_638296?
!dense_136/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_136_638591dense_136_638593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_136_layer_call_and_return_conditional_losses_638309y
IdentityIdentity*dense_136/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????>?
NoOpNoOp"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^dense_134/StatefulPartitionedCall"^dense_135/StatefulPartitionedCall"^dense_136/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!dense_134/StatefulPartitionedCall!dense_134/StatefulPartitionedCall2F
!dense_135/StatefulPartitionedCall!dense_135/StatefulPartitionedCall2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_50_input
?
b
F__inference_flatten_17_layer_call_and_return_conditional_losses_638255

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?1
?
I__inference_sequential_41_layer_call_and_return_conditional_losses_638636
conv2d_50_input*
conv2d_50_638600: 
conv2d_50_638602: *
conv2d_51_638606: @
conv2d_51_638608:@+
conv2d_52_638612:@?
conv2d_52_638614:	?#
dense_134_638619:	?@
dense_134_638621:@#
dense_135_638624:	@?
dense_135_638626:	?#
dense_136_638630:	?>
dense_136_638632:>
identity??!conv2d_50/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!dense_134/StatefulPartitionedCall?!dense_135/StatefulPartitionedCall?!dense_136/StatefulPartitionedCall?"dropout_23/StatefulPartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCallconv2d_50_inputconv2d_50_638600conv2d_50_638602*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_638206?
 max_pooling2d_48/PartitionedCallPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_638161?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0conv2d_51_638606conv2d_51_638608*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_638224?
 max_pooling2d_49/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_638173?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_52_638612conv2d_52_638614*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_638242?
 max_pooling2d_50/PartitionedCallPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_638185?
flatten_17/PartitionedCallPartitionedCall)max_pooling2d_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_638255?
!dense_134/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_134_638619dense_134_638621*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_134_layer_call_and_return_conditional_losses_638268?
!dense_135/StatefulPartitionedCallStatefulPartitionedCall*dense_134/StatefulPartitionedCall:output:0dense_135_638624dense_135_638626*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_135_layer_call_and_return_conditional_losses_638285?
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall*dense_135/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_638373?
!dense_136/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_136_638630dense_136_638632*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_136_layer_call_and_return_conditional_losses_638309y
IdentityIdentity*dense_136/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????>?
NoOpNoOp"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^dense_134/StatefulPartitionedCall"^dense_135/StatefulPartitionedCall"^dense_136/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!dense_134/StatefulPartitionedCall!dense_134/StatefulPartitionedCall2F
!dense_135/StatefulPartitionedCall!dense_135/StatefulPartitionedCall2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_50_input
?/
?
I__inference_sequential_41_layer_call_and_return_conditional_losses_638316

inputs*
conv2d_50_638207: 
conv2d_50_638209: *
conv2d_51_638225: @
conv2d_51_638227:@+
conv2d_52_638243:@?
conv2d_52_638245:	?#
dense_134_638269:	?@
dense_134_638271:@#
dense_135_638286:	@?
dense_135_638288:	?#
dense_136_638310:	?>
dense_136_638312:>
identity??!conv2d_50/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!dense_134/StatefulPartitionedCall?!dense_135/StatefulPartitionedCall?!dense_136/StatefulPartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_50_638207conv2d_50_638209*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_638206?
 max_pooling2d_48/PartitionedCallPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_638161?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0conv2d_51_638225conv2d_51_638227*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_638224?
 max_pooling2d_49/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_638173?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_52_638243conv2d_52_638245*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_638242?
 max_pooling2d_50/PartitionedCallPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_638185?
flatten_17/PartitionedCallPartitionedCall)max_pooling2d_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_638255?
!dense_134/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_134_638269dense_134_638271*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_134_layer_call_and_return_conditional_losses_638268?
!dense_135/StatefulPartitionedCallStatefulPartitionedCall*dense_134/StatefulPartitionedCall:output:0dense_135_638286dense_135_638288*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_135_layer_call_and_return_conditional_losses_638285?
dropout_23/PartitionedCallPartitionedCall*dense_135/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_638296?
!dense_136/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_136_638310dense_136_638312*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_136_layer_call_and_return_conditional_losses_638309y
IdentityIdentity*dense_136/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????>?
NoOpNoOp"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^dense_134/StatefulPartitionedCall"^dense_135/StatefulPartitionedCall"^dense_136/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!dense_134/StatefulPartitionedCall!dense_134/StatefulPartitionedCall2F
!dense_135/StatefulPartitionedCall!dense_135/StatefulPartitionedCall2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_17_layer_call_and_return_conditional_losses_638943

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_49_layer_call_fn_638897

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_638173?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_23_layer_call_fn_638988

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_638296a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_638224

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_50_layer_call_fn_638927

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_638185?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_638296

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_638161

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_638673
conv2d_50_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:	@?
	unknown_8:	?
	unknown_9:	?>

unknown_10:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_50_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_638152o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_50_input
?
?
.__inference_sequential_41_layer_call_fn_638343
conv2d_50_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:	@?
	unknown_8:	?
	unknown_9:	?>

unknown_10:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_50_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_41_layer_call_and_return_conditional_losses_638316o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_50_input
?

?
E__inference_dense_135_layer_call_and_return_conditional_losses_638285

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_dense_136_layer_call_fn_639019

inputs
unknown:	?>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_136_layer_call_and_return_conditional_losses_638309o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_50_layer_call_and_return_conditional_losses_638862

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_638892

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
E__inference_dense_134_layer_call_and_return_conditional_losses_638963

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
__inference__traced_save_639188
file_prefix/
+savev2_conv2d_50_kernel_read_readvariableop-
)savev2_conv2d_50_bias_read_readvariableop/
+savev2_conv2d_51_kernel_read_readvariableop-
)savev2_conv2d_51_bias_read_readvariableop/
+savev2_conv2d_52_kernel_read_readvariableop-
)savev2_conv2d_52_bias_read_readvariableop/
+savev2_dense_134_kernel_read_readvariableop-
)savev2_dense_134_bias_read_readvariableop/
+savev2_dense_135_kernel_read_readvariableop-
)savev2_dense_135_bias_read_readvariableop/
+savev2_dense_136_kernel_read_readvariableop-
)savev2_dense_136_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_50_kernel_m_read_readvariableop4
0savev2_adam_conv2d_50_bias_m_read_readvariableop6
2savev2_adam_conv2d_51_kernel_m_read_readvariableop4
0savev2_adam_conv2d_51_bias_m_read_readvariableop6
2savev2_adam_conv2d_52_kernel_m_read_readvariableop4
0savev2_adam_conv2d_52_bias_m_read_readvariableop6
2savev2_adam_dense_134_kernel_m_read_readvariableop4
0savev2_adam_dense_134_bias_m_read_readvariableop6
2savev2_adam_dense_135_kernel_m_read_readvariableop4
0savev2_adam_dense_135_bias_m_read_readvariableop6
2savev2_adam_dense_136_kernel_m_read_readvariableop4
0savev2_adam_dense_136_bias_m_read_readvariableop6
2savev2_adam_conv2d_50_kernel_v_read_readvariableop4
0savev2_adam_conv2d_50_bias_v_read_readvariableop6
2savev2_adam_conv2d_51_kernel_v_read_readvariableop4
0savev2_adam_conv2d_51_bias_v_read_readvariableop6
2savev2_adam_conv2d_52_kernel_v_read_readvariableop4
0savev2_adam_conv2d_52_bias_v_read_readvariableop6
2savev2_adam_dense_134_kernel_v_read_readvariableop4
0savev2_adam_dense_134_bias_v_read_readvariableop6
2savev2_adam_dense_135_kernel_v_read_readvariableop4
0savev2_adam_dense_135_bias_v_read_readvariableop6
2savev2_adam_dense_136_kernel_v_read_readvariableop4
0savev2_adam_dense_136_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_50_kernel_read_readvariableop)savev2_conv2d_50_bias_read_readvariableop+savev2_conv2d_51_kernel_read_readvariableop)savev2_conv2d_51_bias_read_readvariableop+savev2_conv2d_52_kernel_read_readvariableop)savev2_conv2d_52_bias_read_readvariableop+savev2_dense_134_kernel_read_readvariableop)savev2_dense_134_bias_read_readvariableop+savev2_dense_135_kernel_read_readvariableop)savev2_dense_135_bias_read_readvariableop+savev2_dense_136_kernel_read_readvariableop)savev2_dense_136_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_50_kernel_m_read_readvariableop0savev2_adam_conv2d_50_bias_m_read_readvariableop2savev2_adam_conv2d_51_kernel_m_read_readvariableop0savev2_adam_conv2d_51_bias_m_read_readvariableop2savev2_adam_conv2d_52_kernel_m_read_readvariableop0savev2_adam_conv2d_52_bias_m_read_readvariableop2savev2_adam_dense_134_kernel_m_read_readvariableop0savev2_adam_dense_134_bias_m_read_readvariableop2savev2_adam_dense_135_kernel_m_read_readvariableop0savev2_adam_dense_135_bias_m_read_readvariableop2savev2_adam_dense_136_kernel_m_read_readvariableop0savev2_adam_dense_136_bias_m_read_readvariableop2savev2_adam_conv2d_50_kernel_v_read_readvariableop0savev2_adam_conv2d_50_bias_v_read_readvariableop2savev2_adam_conv2d_51_kernel_v_read_readvariableop0savev2_adam_conv2d_51_bias_v_read_readvariableop2savev2_adam_conv2d_52_kernel_v_read_readvariableop0savev2_adam_conv2d_52_bias_v_read_readvariableop2savev2_adam_dense_134_kernel_v_read_readvariableop0savev2_adam_dense_134_bias_v_read_readvariableop2savev2_adam_dense_135_kernel_v_read_readvariableop0savev2_adam_dense_135_bias_v_read_readvariableop2savev2_adam_dense_136_kernel_v_read_readvariableop0savev2_adam_dense_136_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:@?:?:	?@:@:	@?:?:	?>:>: : : : : : : : : : : : @:@:@?:?:	?@:@:	@?:?:	?>:>: : : @:@:@?:?:	?@:@:	@?:?:	?>:>: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:%	!

_output_shapes
:	@?:!


_output_shapes	
:?:%!

_output_shapes
:	?>: 

_output_shapes
:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:% !

_output_shapes
:	?>: !

_output_shapes
:>:,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: @: %

_output_shapes
:@:-&)
'
_output_shapes
:@?:!'

_output_shapes	
:?:%(!

_output_shapes
:	?@: )

_output_shapes
:@:%*!

_output_shapes
:	@?:!+

_output_shapes	
:?:%,!

_output_shapes
:	?>: -

_output_shapes
:>:.

_output_shapes
: 
?
h
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_638932

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_flatten_17_layer_call_fn_638937

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_638255a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_135_layer_call_fn_638972

inputs
unknown:	@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_135_layer_call_and_return_conditional_losses_638285p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_638872

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_639333
file_prefix;
!assignvariableop_conv2d_50_kernel: /
!assignvariableop_1_conv2d_50_bias: =
#assignvariableop_2_conv2d_51_kernel: @/
!assignvariableop_3_conv2d_51_bias:@>
#assignvariableop_4_conv2d_52_kernel:@?0
!assignvariableop_5_conv2d_52_bias:	?6
#assignvariableop_6_dense_134_kernel:	?@/
!assignvariableop_7_dense_134_bias:@6
#assignvariableop_8_dense_135_kernel:	@?0
!assignvariableop_9_dense_135_bias:	?7
$assignvariableop_10_dense_136_kernel:	?>0
"assignvariableop_11_dense_136_bias:>'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: #
assignvariableop_19_total: #
assignvariableop_20_count: E
+assignvariableop_21_adam_conv2d_50_kernel_m: 7
)assignvariableop_22_adam_conv2d_50_bias_m: E
+assignvariableop_23_adam_conv2d_51_kernel_m: @7
)assignvariableop_24_adam_conv2d_51_bias_m:@F
+assignvariableop_25_adam_conv2d_52_kernel_m:@?8
)assignvariableop_26_adam_conv2d_52_bias_m:	?>
+assignvariableop_27_adam_dense_134_kernel_m:	?@7
)assignvariableop_28_adam_dense_134_bias_m:@>
+assignvariableop_29_adam_dense_135_kernel_m:	@?8
)assignvariableop_30_adam_dense_135_bias_m:	?>
+assignvariableop_31_adam_dense_136_kernel_m:	?>7
)assignvariableop_32_adam_dense_136_bias_m:>E
+assignvariableop_33_adam_conv2d_50_kernel_v: 7
)assignvariableop_34_adam_conv2d_50_bias_v: E
+assignvariableop_35_adam_conv2d_51_kernel_v: @7
)assignvariableop_36_adam_conv2d_51_bias_v:@F
+assignvariableop_37_adam_conv2d_52_kernel_v:@?8
)assignvariableop_38_adam_conv2d_52_bias_v:	?>
+assignvariableop_39_adam_dense_134_kernel_v:	?@7
)assignvariableop_40_adam_dense_134_bias_v:@>
+assignvariableop_41_adam_dense_135_kernel_v:	@?8
)assignvariableop_42_adam_dense_135_bias_v:	?>
+assignvariableop_43_adam_dense_136_kernel_v:	?>7
)assignvariableop_44_adam_dense_136_bias_v:>
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_50_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_50_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_51_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_51_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_52_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_52_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_134_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_134_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_135_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_135_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_136_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_136_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_50_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_50_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_51_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_51_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_52_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_52_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_134_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_134_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_135_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_135_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_136_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_136_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_50_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_50_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_51_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_51_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_52_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_52_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_dense_134_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_134_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_dense_135_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_135_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_dense_136_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_dense_136_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
e
F__inference_dropout_23_layer_call_and_return_conditional_losses_639010

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?N
?
!__inference__wrapped_model_638152
conv2d_50_inputP
6sequential_41_conv2d_50_conv2d_readvariableop_resource: E
7sequential_41_conv2d_50_biasadd_readvariableop_resource: P
6sequential_41_conv2d_51_conv2d_readvariableop_resource: @E
7sequential_41_conv2d_51_biasadd_readvariableop_resource:@Q
6sequential_41_conv2d_52_conv2d_readvariableop_resource:@?F
7sequential_41_conv2d_52_biasadd_readvariableop_resource:	?I
6sequential_41_dense_134_matmul_readvariableop_resource:	?@E
7sequential_41_dense_134_biasadd_readvariableop_resource:@I
6sequential_41_dense_135_matmul_readvariableop_resource:	@?F
7sequential_41_dense_135_biasadd_readvariableop_resource:	?I
6sequential_41_dense_136_matmul_readvariableop_resource:	?>E
7sequential_41_dense_136_biasadd_readvariableop_resource:>
identity??.sequential_41/conv2d_50/BiasAdd/ReadVariableOp?-sequential_41/conv2d_50/Conv2D/ReadVariableOp?.sequential_41/conv2d_51/BiasAdd/ReadVariableOp?-sequential_41/conv2d_51/Conv2D/ReadVariableOp?.sequential_41/conv2d_52/BiasAdd/ReadVariableOp?-sequential_41/conv2d_52/Conv2D/ReadVariableOp?.sequential_41/dense_134/BiasAdd/ReadVariableOp?-sequential_41/dense_134/MatMul/ReadVariableOp?.sequential_41/dense_135/BiasAdd/ReadVariableOp?-sequential_41/dense_135/MatMul/ReadVariableOp?.sequential_41/dense_136/BiasAdd/ReadVariableOp?-sequential_41/dense_136/MatMul/ReadVariableOp?
-sequential_41/conv2d_50/Conv2D/ReadVariableOpReadVariableOp6sequential_41_conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential_41/conv2d_50/Conv2DConv2Dconv2d_50_input5sequential_41/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
.sequential_41/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp7sequential_41_conv2d_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_41/conv2d_50/BiasAddBiasAdd'sequential_41/conv2d_50/Conv2D:output:06sequential_41/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
sequential_41/conv2d_50/ReluRelu(sequential_41/conv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
&sequential_41/max_pooling2d_48/MaxPoolMaxPool*sequential_41/conv2d_50/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
-sequential_41/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6sequential_41_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
sequential_41/conv2d_51/Conv2DConv2D/sequential_41/max_pooling2d_48/MaxPool:output:05sequential_41/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
.sequential_41/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7sequential_41_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_41/conv2d_51/BiasAddBiasAdd'sequential_41/conv2d_51/Conv2D:output:06sequential_41/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
sequential_41/conv2d_51/ReluRelu(sequential_41/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
&sequential_41/max_pooling2d_49/MaxPoolMaxPool*sequential_41/conv2d_51/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
-sequential_41/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6sequential_41_conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
sequential_41/conv2d_52/Conv2DConv2D/sequential_41/max_pooling2d_49/MaxPool:output:05sequential_41/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
.sequential_41/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7sequential_41_conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_41/conv2d_52/BiasAddBiasAdd'sequential_41/conv2d_52/Conv2D:output:06sequential_41/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
sequential_41/conv2d_52/ReluRelu(sequential_41/conv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
&sequential_41/max_pooling2d_50/MaxPoolMaxPool*sequential_41/conv2d_52/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
o
sequential_41/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
 sequential_41/flatten_17/ReshapeReshape/sequential_41/max_pooling2d_50/MaxPool:output:0'sequential_41/flatten_17/Const:output:0*
T0*(
_output_shapes
:???????????
-sequential_41/dense_134/MatMul/ReadVariableOpReadVariableOp6sequential_41_dense_134_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
sequential_41/dense_134/MatMulMatMul)sequential_41/flatten_17/Reshape:output:05sequential_41/dense_134/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
.sequential_41/dense_134/BiasAdd/ReadVariableOpReadVariableOp7sequential_41_dense_134_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_41/dense_134/BiasAddBiasAdd(sequential_41/dense_134/MatMul:product:06sequential_41/dense_134/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
sequential_41/dense_134/ReluRelu(sequential_41/dense_134/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
-sequential_41/dense_135/MatMul/ReadVariableOpReadVariableOp6sequential_41_dense_135_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
sequential_41/dense_135/MatMulMatMul*sequential_41/dense_134/Relu:activations:05sequential_41/dense_135/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
.sequential_41/dense_135/BiasAdd/ReadVariableOpReadVariableOp7sequential_41_dense_135_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_41/dense_135/BiasAddBiasAdd(sequential_41/dense_135/MatMul:product:06sequential_41/dense_135/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential_41/dense_135/ReluRelu(sequential_41/dense_135/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
!sequential_41/dropout_23/IdentityIdentity*sequential_41/dense_135/Relu:activations:0*
T0*(
_output_shapes
:???????????
-sequential_41/dense_136/MatMul/ReadVariableOpReadVariableOp6sequential_41_dense_136_matmul_readvariableop_resource*
_output_shapes
:	?>*
dtype0?
sequential_41/dense_136/MatMulMatMul*sequential_41/dropout_23/Identity:output:05sequential_41/dense_136/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>?
.sequential_41/dense_136/BiasAdd/ReadVariableOpReadVariableOp7sequential_41_dense_136_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype0?
sequential_41/dense_136/BiasAddBiasAdd(sequential_41/dense_136/MatMul:product:06sequential_41/dense_136/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>?
sequential_41/dense_136/SoftmaxSoftmax(sequential_41/dense_136/BiasAdd:output:0*
T0*'
_output_shapes
:?????????>x
IdentityIdentity)sequential_41/dense_136/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????>?
NoOpNoOp/^sequential_41/conv2d_50/BiasAdd/ReadVariableOp.^sequential_41/conv2d_50/Conv2D/ReadVariableOp/^sequential_41/conv2d_51/BiasAdd/ReadVariableOp.^sequential_41/conv2d_51/Conv2D/ReadVariableOp/^sequential_41/conv2d_52/BiasAdd/ReadVariableOp.^sequential_41/conv2d_52/Conv2D/ReadVariableOp/^sequential_41/dense_134/BiasAdd/ReadVariableOp.^sequential_41/dense_134/MatMul/ReadVariableOp/^sequential_41/dense_135/BiasAdd/ReadVariableOp.^sequential_41/dense_135/MatMul/ReadVariableOp/^sequential_41/dense_136/BiasAdd/ReadVariableOp.^sequential_41/dense_136/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2`
.sequential_41/conv2d_50/BiasAdd/ReadVariableOp.sequential_41/conv2d_50/BiasAdd/ReadVariableOp2^
-sequential_41/conv2d_50/Conv2D/ReadVariableOp-sequential_41/conv2d_50/Conv2D/ReadVariableOp2`
.sequential_41/conv2d_51/BiasAdd/ReadVariableOp.sequential_41/conv2d_51/BiasAdd/ReadVariableOp2^
-sequential_41/conv2d_51/Conv2D/ReadVariableOp-sequential_41/conv2d_51/Conv2D/ReadVariableOp2`
.sequential_41/conv2d_52/BiasAdd/ReadVariableOp.sequential_41/conv2d_52/BiasAdd/ReadVariableOp2^
-sequential_41/conv2d_52/Conv2D/ReadVariableOp-sequential_41/conv2d_52/Conv2D/ReadVariableOp2`
.sequential_41/dense_134/BiasAdd/ReadVariableOp.sequential_41/dense_134/BiasAdd/ReadVariableOp2^
-sequential_41/dense_134/MatMul/ReadVariableOp-sequential_41/dense_134/MatMul/ReadVariableOp2`
.sequential_41/dense_135/BiasAdd/ReadVariableOp.sequential_41/dense_135/BiasAdd/ReadVariableOp2^
-sequential_41/dense_135/MatMul/ReadVariableOp-sequential_41/dense_135/MatMul/ReadVariableOp2`
.sequential_41/dense_136/BiasAdd/ReadVariableOp.sequential_41/dense_136/BiasAdd/ReadVariableOp2^
-sequential_41/dense_136/MatMul/ReadVariableOp-sequential_41/dense_136/MatMul/ReadVariableOp:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_50_input
?

?
E__inference_dense_134_layer_call_and_return_conditional_losses_638268

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_48_layer_call_fn_638867

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_638161?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_41_layer_call_fn_638731

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:	@?
	unknown_8:	?
	unknown_9:	?>

unknown_10:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_41_layer_call_and_return_conditional_losses_638502o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_52_layer_call_fn_638911

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_638242x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_dense_134_layer_call_fn_638952

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_134_layer_call_and_return_conditional_losses_638268o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_52_layer_call_and_return_conditional_losses_638242

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_638902

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_638173

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?F
?	
I__inference_sequential_41_layer_call_and_return_conditional_losses_638842

inputsB
(conv2d_50_conv2d_readvariableop_resource: 7
)conv2d_50_biasadd_readvariableop_resource: B
(conv2d_51_conv2d_readvariableop_resource: @7
)conv2d_51_biasadd_readvariableop_resource:@C
(conv2d_52_conv2d_readvariableop_resource:@?8
)conv2d_52_biasadd_readvariableop_resource:	?;
(dense_134_matmul_readvariableop_resource:	?@7
)dense_134_biasadd_readvariableop_resource:@;
(dense_135_matmul_readvariableop_resource:	@?8
)dense_135_biasadd_readvariableop_resource:	?;
(dense_136_matmul_readvariableop_resource:	?>7
)dense_136_biasadd_readvariableop_resource:>
identity?? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp? conv2d_51/BiasAdd/ReadVariableOp?conv2d_51/Conv2D/ReadVariableOp? conv2d_52/BiasAdd/ReadVariableOp?conv2d_52/Conv2D/ReadVariableOp? dense_134/BiasAdd/ReadVariableOp?dense_134/MatMul/ReadVariableOp? dense_135/BiasAdd/ReadVariableOp?dense_135/MatMul/ReadVariableOp? dense_136/BiasAdd/ReadVariableOp?dense_136/MatMul/ReadVariableOp?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_50/Conv2DConv2Dinputs'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
max_pooling2d_48/MaxPoolMaxPoolconv2d_50/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_51/Conv2DConv2D!max_pooling2d_48/MaxPool:output:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
max_pooling2d_49/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_52/Conv2DConv2D!max_pooling2d_49/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
max_pooling2d_50/MaxPoolMaxPoolconv2d_52/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
a
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_17/ReshapeReshape!max_pooling2d_50/MaxPool:output:0flatten_17/Const:output:0*
T0*(
_output_shapes
:???????????
dense_134/MatMul/ReadVariableOpReadVariableOp(dense_134_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_134/MatMulMatMulflatten_17/Reshape:output:0'dense_134/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
 dense_134/BiasAdd/ReadVariableOpReadVariableOp)dense_134_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_134/BiasAddBiasAdddense_134/MatMul:product:0(dense_134/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
dense_134/ReluReludense_134/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_135/MatMul/ReadVariableOpReadVariableOp(dense_135_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
dense_135/MatMulMatMuldense_134/Relu:activations:0'dense_135/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
 dense_135/BiasAdd/ReadVariableOpReadVariableOp)dense_135_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_135/BiasAddBiasAdddense_135/MatMul:product:0(dense_135/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
dense_135/ReluReludense_135/BiasAdd:output:0*
T0*(
_output_shapes
:??????????]
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_23/dropout/MulMuldense_135/Relu:activations:0!dropout_23/dropout/Const:output:0*
T0*(
_output_shapes
:??????????d
dropout_23/dropout/ShapeShapedense_135/Relu:activations:0*
T0*
_output_shapes
:?
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0f
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_136/MatMul/ReadVariableOpReadVariableOp(dense_136_matmul_readvariableop_resource*
_output_shapes
:	?>*
dtype0?
dense_136/MatMulMatMuldropout_23/dropout/Mul_1:z:0'dense_136/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>?
 dense_136/BiasAdd/ReadVariableOpReadVariableOp)dense_136_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype0?
dense_136/BiasAddBiasAdddense_136/MatMul:product:0(dense_136/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>j
dense_136/SoftmaxSoftmaxdense_136/BiasAdd:output:0*
T0*'
_output_shapes
:?????????>j
IdentityIdentitydense_136/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????>?
NoOpNoOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^dense_134/BiasAdd/ReadVariableOp ^dense_134/MatMul/ReadVariableOp!^dense_135/BiasAdd/ReadVariableOp ^dense_135/MatMul/ReadVariableOp!^dense_136/BiasAdd/ReadVariableOp ^dense_136/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 dense_134/BiasAdd/ReadVariableOp dense_134/BiasAdd/ReadVariableOp2B
dense_134/MatMul/ReadVariableOpdense_134/MatMul/ReadVariableOp2D
 dense_135/BiasAdd/ReadVariableOp dense_135/BiasAdd/ReadVariableOp2B
dense_135/MatMul/ReadVariableOpdense_135/MatMul/ReadVariableOp2D
 dense_136/BiasAdd/ReadVariableOp dense_136/BiasAdd/ReadVariableOp2B
dense_136/MatMul/ReadVariableOpdense_136/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?	
I__inference_sequential_41_layer_call_and_return_conditional_losses_638783

inputsB
(conv2d_50_conv2d_readvariableop_resource: 7
)conv2d_50_biasadd_readvariableop_resource: B
(conv2d_51_conv2d_readvariableop_resource: @7
)conv2d_51_biasadd_readvariableop_resource:@C
(conv2d_52_conv2d_readvariableop_resource:@?8
)conv2d_52_biasadd_readvariableop_resource:	?;
(dense_134_matmul_readvariableop_resource:	?@7
)dense_134_biasadd_readvariableop_resource:@;
(dense_135_matmul_readvariableop_resource:	@?8
)dense_135_biasadd_readvariableop_resource:	?;
(dense_136_matmul_readvariableop_resource:	?>7
)dense_136_biasadd_readvariableop_resource:>
identity?? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp? conv2d_51/BiasAdd/ReadVariableOp?conv2d_51/Conv2D/ReadVariableOp? conv2d_52/BiasAdd/ReadVariableOp?conv2d_52/Conv2D/ReadVariableOp? dense_134/BiasAdd/ReadVariableOp?dense_134/MatMul/ReadVariableOp? dense_135/BiasAdd/ReadVariableOp?dense_135/MatMul/ReadVariableOp? dense_136/BiasAdd/ReadVariableOp?dense_136/MatMul/ReadVariableOp?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_50/Conv2DConv2Dinputs'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
max_pooling2d_48/MaxPoolMaxPoolconv2d_50/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_51/Conv2DConv2D!max_pooling2d_48/MaxPool:output:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
max_pooling2d_49/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_52/Conv2DConv2D!max_pooling2d_49/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
max_pooling2d_50/MaxPoolMaxPoolconv2d_52/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
a
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_17/ReshapeReshape!max_pooling2d_50/MaxPool:output:0flatten_17/Const:output:0*
T0*(
_output_shapes
:???????????
dense_134/MatMul/ReadVariableOpReadVariableOp(dense_134_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_134/MatMulMatMulflatten_17/Reshape:output:0'dense_134/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
 dense_134/BiasAdd/ReadVariableOpReadVariableOp)dense_134_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_134/BiasAddBiasAdddense_134/MatMul:product:0(dense_134/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
dense_134/ReluReludense_134/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_135/MatMul/ReadVariableOpReadVariableOp(dense_135_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
dense_135/MatMulMatMuldense_134/Relu:activations:0'dense_135/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
 dense_135/BiasAdd/ReadVariableOpReadVariableOp)dense_135_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_135/BiasAddBiasAdddense_135/MatMul:product:0(dense_135/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
dense_135/ReluReludense_135/BiasAdd:output:0*
T0*(
_output_shapes
:??????????p
dropout_23/IdentityIdentitydense_135/Relu:activations:0*
T0*(
_output_shapes
:???????????
dense_136/MatMul/ReadVariableOpReadVariableOp(dense_136_matmul_readvariableop_resource*
_output_shapes
:	?>*
dtype0?
dense_136/MatMulMatMuldropout_23/Identity:output:0'dense_136/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>?
 dense_136/BiasAdd/ReadVariableOpReadVariableOp)dense_136_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype0?
dense_136/BiasAddBiasAdddense_136/MatMul:product:0(dense_136/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>j
dense_136/SoftmaxSoftmaxdense_136/BiasAdd:output:0*
T0*'
_output_shapes
:?????????>j
IdentityIdentitydense_136/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????>?
NoOpNoOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^dense_134/BiasAdd/ReadVariableOp ^dense_134/MatMul/ReadVariableOp!^dense_135/BiasAdd/ReadVariableOp ^dense_135/MatMul/ReadVariableOp!^dense_136/BiasAdd/ReadVariableOp ^dense_136/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 dense_134/BiasAdd/ReadVariableOp dense_134/BiasAdd/ReadVariableOp2B
dense_134/MatMul/ReadVariableOpdense_134/MatMul/ReadVariableOp2D
 dense_135/BiasAdd/ReadVariableOp dense_135/BiasAdd/ReadVariableOp2B
dense_135/MatMul/ReadVariableOpdense_135/MatMul/ReadVariableOp2D
 dense_136/BiasAdd/ReadVariableOp dense_136/BiasAdd/ReadVariableOp2B
dense_136/MatMul/ReadVariableOpdense_136/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_136_layer_call_and_return_conditional_losses_639030

inputs1
matmul_readvariableop_resource:	?>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?>*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????>`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????>w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_dense_136_layer_call_and_return_conditional_losses_638309

inputs1
matmul_readvariableop_resource:	?>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?>*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????>`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????>w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?1
?
I__inference_sequential_41_layer_call_and_return_conditional_losses_638502

inputs*
conv2d_50_638466: 
conv2d_50_638468: *
conv2d_51_638472: @
conv2d_51_638474:@+
conv2d_52_638478:@?
conv2d_52_638480:	?#
dense_134_638485:	?@
dense_134_638487:@#
dense_135_638490:	@?
dense_135_638492:	?#
dense_136_638496:	?>
dense_136_638498:>
identity??!conv2d_50/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_52/StatefulPartitionedCall?!dense_134/StatefulPartitionedCall?!dense_135/StatefulPartitionedCall?!dense_136/StatefulPartitionedCall?"dropout_23/StatefulPartitionedCall?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_50_638466conv2d_50_638468*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_638206?
 max_pooling2d_48/PartitionedCallPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_638161?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0conv2d_51_638472conv2d_51_638474*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_638224?
 max_pooling2d_49/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_638173?
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_52_638478conv2d_52_638480*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_638242?
 max_pooling2d_50/PartitionedCallPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_638185?
flatten_17/PartitionedCallPartitionedCall)max_pooling2d_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_17_layer_call_and_return_conditional_losses_638255?
!dense_134/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dense_134_638485dense_134_638487*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_134_layer_call_and_return_conditional_losses_638268?
!dense_135/StatefulPartitionedCallStatefulPartitionedCall*dense_134/StatefulPartitionedCall:output:0dense_135_638490dense_135_638492*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_135_layer_call_and_return_conditional_losses_638285?
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall*dense_135/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_638373?
!dense_136/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_136_638496dense_136_638498*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_136_layer_call_and_return_conditional_losses_638309y
IdentityIdentity*dense_136/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????>?
NoOpNoOp"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^dense_134/StatefulPartitionedCall"^dense_135/StatefulPartitionedCall"^dense_136/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!dense_134/StatefulPartitionedCall!dense_134/StatefulPartitionedCall2F
!dense_135/StatefulPartitionedCall!dense_135/StatefulPartitionedCall2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_638185

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_51_layer_call_fn_638881

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_638224w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
.__inference_sequential_41_layer_call_fn_638558
conv2d_50_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:	@?
	unknown_8:	?
	unknown_9:	?>

unknown_10:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_50_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_41_layer_call_and_return_conditional_losses_638502o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_50_input
?
d
+__inference_dropout_23_layer_call_fn_638993

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_638373p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_52_layer_call_and_return_conditional_losses_638922

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
.__inference_sequential_41_layer_call_fn_638702

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:	@?
	unknown_8:	?
	unknown_9:	?>

unknown_10:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_41_layer_call_and_return_conditional_losses_638316o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
e
F__inference_dropout_23_layer_call_and_return_conditional_losses_638373

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_dense_135_layer_call_and_return_conditional_losses_638983

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_conv2d_50_layer_call_fn_638851

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_50_layer_call_and_return_conditional_losses_638206w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
conv2d_50_input@
!serving_default_conv2d_50_input:0?????????=
	dense_1360
StatefulPartitionedCall:0?????????>tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
?
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias
 ,_jit_compiled_convolution_op"
_tf_keras_layer
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
?
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias
 ;_jit_compiled_convolution_op"
_tf_keras_layer
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
?
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
?
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias"
_tf_keras_layer
?
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias"
_tf_keras_layer
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator"
_tf_keras_layer
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

ekernel
fbias"
_tf_keras_layer
v
0
1
*2
+3
94
:5
N6
O7
V8
W9
e10
f11"
trackable_list_wrapper
v
0
1
*2
+3
94
:5
N6
O7
V8
W9
e10
f11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
ltrace_0
mtrace_1
ntrace_2
otrace_32?
.__inference_sequential_41_layer_call_fn_638343
.__inference_sequential_41_layer_call_fn_638702
.__inference_sequential_41_layer_call_fn_638731
.__inference_sequential_41_layer_call_fn_638558?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zltrace_0zmtrace_1zntrace_2zotrace_3
?
ptrace_0
qtrace_1
rtrace_2
strace_32?
I__inference_sequential_41_layer_call_and_return_conditional_losses_638783
I__inference_sequential_41_layer_call_and_return_conditional_losses_638842
I__inference_sequential_41_layer_call_and_return_conditional_losses_638597
I__inference_sequential_41_layer_call_and_return_conditional_losses_638636?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zptrace_0zqtrace_1zrtrace_2zstrace_3
?B?
!__inference__wrapped_model_638152conv2d_50_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
titer

ubeta_1

vbeta_2
	wdecay
xlearning_ratem?m?*m?+m?9m?:m?Nm?Om?Vm?Wm?em?fm?v?v?*v?+v?9v?:v?Nv?Ov?Vv?Wv?ev?fv?"
	optimizer
,
yserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
trace_02?
*__inference_conv2d_50_layer_call_fn_638851?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ztrace_0
?
?trace_02?
E__inference_conv2d_50_layer_call_and_return_conditional_losses_638862?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:( 2conv2d_50/kernel
: 2conv2d_50/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_max_pooling2d_48_layer_call_fn_638867?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_638872?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_conv2d_51_layer_call_fn_638881?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_638892?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:( @2conv2d_51/kernel
:@2conv2d_51/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_max_pooling2d_49_layer_call_fn_638897?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_638902?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_conv2d_52_layer_call_fn_638911?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_conv2d_52_layer_call_and_return_conditional_losses_638922?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
+:)@?2conv2d_52/kernel
:?2conv2d_52/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_max_pooling2d_50_layer_call_fn_638927?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_638932?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_flatten_17_layer_call_fn_638937?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_flatten_17_layer_call_and_return_conditional_losses_638943?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_dense_134_layer_call_fn_638952?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_dense_134_layer_call_and_return_conditional_losses_638963?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
#:!	?@2dense_134/kernel
:@2dense_134/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_dense_135_layer_call_fn_638972?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_dense_135_layer_call_and_return_conditional_losses_638983?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
#:!	@?2dense_135/kernel
:?2dense_135/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
+__inference_dropout_23_layer_call_fn_638988
+__inference_dropout_23_layer_call_fn_638993?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
F__inference_dropout_23_layer_call_and_return_conditional_losses_638998
F__inference_dropout_23_layer_call_and_return_conditional_losses_639010?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_dense_136_layer_call_fn_639019?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_dense_136_layer_call_and_return_conditional_losses_639030?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
#:!	?>2dense_136/kernel
:>2dense_136/bias
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_sequential_41_layer_call_fn_638343conv2d_50_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
.__inference_sequential_41_layer_call_fn_638702inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
.__inference_sequential_41_layer_call_fn_638731inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
.__inference_sequential_41_layer_call_fn_638558conv2d_50_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_sequential_41_layer_call_and_return_conditional_losses_638783inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_sequential_41_layer_call_and_return_conditional_losses_638842inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_sequential_41_layer_call_and_return_conditional_losses_638597conv2d_50_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_sequential_41_layer_call_and_return_conditional_losses_638636conv2d_50_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
$__inference_signature_wrapper_638673conv2d_50_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_conv2d_50_layer_call_fn_638851inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_conv2d_50_layer_call_and_return_conditional_losses_638862inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
1__inference_max_pooling2d_48_layer_call_fn_638867inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_638872inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_conv2d_51_layer_call_fn_638881inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_638892inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
1__inference_max_pooling2d_49_layer_call_fn_638897inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_638902inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_conv2d_52_layer_call_fn_638911inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_conv2d_52_layer_call_and_return_conditional_losses_638922inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
1__inference_max_pooling2d_50_layer_call_fn_638927inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_638932inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_flatten_17_layer_call_fn_638937inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_flatten_17_layer_call_and_return_conditional_losses_638943inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_dense_134_layer_call_fn_638952inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dense_134_layer_call_and_return_conditional_losses_638963inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_dense_135_layer_call_fn_638972inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dense_135_layer_call_and_return_conditional_losses_638983inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_dropout_23_layer_call_fn_638988inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
+__inference_dropout_23_layer_call_fn_638993inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_dropout_23_layer_call_and_return_conditional_losses_638998inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_dropout_23_layer_call_and_return_conditional_losses_639010inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_dense_136_layer_call_fn_639019inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dense_136_layer_call_and_return_conditional_losses_639030inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:- 2Adam/conv2d_50/kernel/m
!: 2Adam/conv2d_50/bias/m
/:- @2Adam/conv2d_51/kernel/m
!:@2Adam/conv2d_51/bias/m
0:.@?2Adam/conv2d_52/kernel/m
": ?2Adam/conv2d_52/bias/m
(:&	?@2Adam/dense_134/kernel/m
!:@2Adam/dense_134/bias/m
(:&	@?2Adam/dense_135/kernel/m
": ?2Adam/dense_135/bias/m
(:&	?>2Adam/dense_136/kernel/m
!:>2Adam/dense_136/bias/m
/:- 2Adam/conv2d_50/kernel/v
!: 2Adam/conv2d_50/bias/v
/:- @2Adam/conv2d_51/kernel/v
!:@2Adam/conv2d_51/bias/v
0:.@?2Adam/conv2d_52/kernel/v
": ?2Adam/conv2d_52/bias/v
(:&	?@2Adam/dense_134/kernel/v
!:@2Adam/dense_134/bias/v
(:&	@?2Adam/dense_135/kernel/v
": ?2Adam/dense_135/bias/v
(:&	?>2Adam/dense_136/kernel/v
!:>2Adam/dense_136/bias/v?
!__inference__wrapped_model_638152?*+9:NOVWef@?=
6?3
1?.
conv2d_50_input?????????
? "5?2
0
	dense_136#? 
	dense_136?????????>?
E__inference_conv2d_50_layer_call_and_return_conditional_losses_638862l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_50_layer_call_fn_638851_7?4
-?*
(?%
inputs?????????
? " ?????????? ?
E__inference_conv2d_51_layer_call_and_return_conditional_losses_638892l*+7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_51_layer_call_fn_638881_*+7?4
-?*
(?%
inputs????????? 
? " ??????????@?
E__inference_conv2d_52_layer_call_and_return_conditional_losses_638922m9:7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_52_layer_call_fn_638911`9:7?4
-?*
(?%
inputs?????????@
? "!????????????
E__inference_dense_134_layer_call_and_return_conditional_losses_638963]NO0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? ~
*__inference_dense_134_layer_call_fn_638952PNO0?-
&?#
!?
inputs??????????
? "??????????@?
E__inference_dense_135_layer_call_and_return_conditional_losses_638983]VW/?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? ~
*__inference_dense_135_layer_call_fn_638972PVW/?,
%?"
 ?
inputs?????????@
? "????????????
E__inference_dense_136_layer_call_and_return_conditional_losses_639030]ef0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????>
? ~
*__inference_dense_136_layer_call_fn_639019Pef0?-
&?#
!?
inputs??????????
? "??????????>?
F__inference_dropout_23_layer_call_and_return_conditional_losses_638998^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
F__inference_dropout_23_layer_call_and_return_conditional_losses_639010^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
+__inference_dropout_23_layer_call_fn_638988Q4?1
*?'
!?
inputs??????????
p 
? "????????????
+__inference_dropout_23_layer_call_fn_638993Q4?1
*?'
!?
inputs??????????
p
? "????????????
F__inference_flatten_17_layer_call_and_return_conditional_losses_638943b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_flatten_17_layer_call_fn_638937U8?5
.?+
)?&
inputs??????????
? "????????????
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_638872?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_48_layer_call_fn_638867?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_638902?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_49_layer_call_fn_638897?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_638932?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_50_layer_call_fn_638927?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_sequential_41_layer_call_and_return_conditional_losses_638597*+9:NOVWefH?E
>?;
1?.
conv2d_50_input?????????
p 

 
? "%?"
?
0?????????>
? ?
I__inference_sequential_41_layer_call_and_return_conditional_losses_638636*+9:NOVWefH?E
>?;
1?.
conv2d_50_input?????????
p

 
? "%?"
?
0?????????>
? ?
I__inference_sequential_41_layer_call_and_return_conditional_losses_638783v*+9:NOVWef??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????>
? ?
I__inference_sequential_41_layer_call_and_return_conditional_losses_638842v*+9:NOVWef??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????>
? ?
.__inference_sequential_41_layer_call_fn_638343r*+9:NOVWefH?E
>?;
1?.
conv2d_50_input?????????
p 

 
? "??????????>?
.__inference_sequential_41_layer_call_fn_638558r*+9:NOVWefH?E
>?;
1?.
conv2d_50_input?????????
p

 
? "??????????>?
.__inference_sequential_41_layer_call_fn_638702i*+9:NOVWef??<
5?2
(?%
inputs?????????
p 

 
? "??????????>?
.__inference_sequential_41_layer_call_fn_638731i*+9:NOVWef??<
5?2
(?%
inputs?????????
p

 
? "??????????>?
$__inference_signature_wrapper_638673?*+9:NOVWefS?P
? 
I?F
D
conv2d_50_input1?.
conv2d_50_input?????????"5?2
0
	dense_136#? 
	dense_136?????????>