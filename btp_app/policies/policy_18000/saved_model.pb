мс
ы

ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
М
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

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
њ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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

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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
>
Minimum
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02v2.7.0-rc1-69-gc256c071bb28Ъ
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	

sequential_3/conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namesequential_3/conv2d_12/kernel

1sequential_3/conv2d_12/kernel/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_12/kernel*'
_output_shapes
:*
dtype0

sequential_3/conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namesequential_3/conv2d_12/bias

/sequential_3/conv2d_12/bias/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_12/bias*
_output_shapes	
:*
dtype0
З
/sequential_3/batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/sequential_3/batch_normalization_12/moving_mean
А
Csequential_3/batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp/sequential_3/batch_normalization_12/moving_mean*
_output_shapes	
:*
dtype0
П
3sequential_3/batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53sequential_3/batch_normalization_12/moving_variance
И
Gsequential_3/batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp3sequential_3/batch_normalization_12/moving_variance*
_output_shapes	
:*
dtype0
Ћ
)sequential_3/batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)sequential_3/batch_normalization_12/gamma
Є
=sequential_3/batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOp)sequential_3/batch_normalization_12/gamma*
_output_shapes	
:*
dtype0
Љ
(sequential_3/batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(sequential_3/batch_normalization_12/beta
Ђ
<sequential_3/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOp(sequential_3/batch_normalization_12/beta*
_output_shapes	
:*
dtype0
 
sequential_3/conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namesequential_3/conv2d_13/kernel

1sequential_3/conv2d_13/kernel/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_13/kernel*(
_output_shapes
:*
dtype0

sequential_3/conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namesequential_3/conv2d_13/bias

/sequential_3/conv2d_13/bias/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_13/bias*
_output_shapes	
:*
dtype0
З
/sequential_3/batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/sequential_3/batch_normalization_13/moving_mean
А
Csequential_3/batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp/sequential_3/batch_normalization_13/moving_mean*
_output_shapes	
:*
dtype0
П
3sequential_3/batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53sequential_3/batch_normalization_13/moving_variance
И
Gsequential_3/batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp3sequential_3/batch_normalization_13/moving_variance*
_output_shapes	
:*
dtype0
Ћ
)sequential_3/batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)sequential_3/batch_normalization_13/gamma
Є
=sequential_3/batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOp)sequential_3/batch_normalization_13/gamma*
_output_shapes	
:*
dtype0
Љ
(sequential_3/batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(sequential_3/batch_normalization_13/beta
Ђ
<sequential_3/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOp(sequential_3/batch_normalization_13/beta*
_output_shapes	
:*
dtype0

sequential_3/conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namesequential_3/conv2d_14/kernel

1sequential_3/conv2d_14/kernel/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_14/kernel*'
_output_shapes
:@*
dtype0

sequential_3/conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namesequential_3/conv2d_14/bias

/sequential_3/conv2d_14/bias/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_14/bias*
_output_shapes
:@*
dtype0
Ж
/sequential_3/batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/sequential_3/batch_normalization_14/moving_mean
Џ
Csequential_3/batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp/sequential_3/batch_normalization_14/moving_mean*
_output_shapes
:@*
dtype0
О
3sequential_3/batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53sequential_3/batch_normalization_14/moving_variance
З
Gsequential_3/batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp3sequential_3/batch_normalization_14/moving_variance*
_output_shapes
:@*
dtype0
Њ
)sequential_3/batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)sequential_3/batch_normalization_14/gamma
Ѓ
=sequential_3/batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOp)sequential_3/batch_normalization_14/gamma*
_output_shapes
:@*
dtype0
Ј
(sequential_3/batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(sequential_3/batch_normalization_14/beta
Ё
<sequential_3/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOp(sequential_3/batch_normalization_14/beta*
_output_shapes
:@*
dtype0

sequential_3/conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *.
shared_namesequential_3/conv2d_15/kernel

1sequential_3/conv2d_15/kernel/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_15/kernel*&
_output_shapes
:@ *
dtype0

sequential_3/conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namesequential_3/conv2d_15/bias

/sequential_3/conv2d_15/bias/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_15/bias*
_output_shapes
: *
dtype0
Ж
/sequential_3/batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/sequential_3/batch_normalization_15/moving_mean
Џ
Csequential_3/batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp/sequential_3/batch_normalization_15/moving_mean*
_output_shapes
: *
dtype0
О
3sequential_3/batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53sequential_3/batch_normalization_15/moving_variance
З
Gsequential_3/batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp3sequential_3/batch_normalization_15/moving_variance*
_output_shapes
: *
dtype0
Њ
)sequential_3/batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)sequential_3/batch_normalization_15/gamma
Ѓ
=sequential_3/batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOp)sequential_3/batch_normalization_15/gamma*
_output_shapes
: *
dtype0
Ј
(sequential_3/batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(sequential_3/batch_normalization_15/beta
Ё
<sequential_3/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOp(sequential_3/batch_normalization_15/beta*
_output_shapes
: *
dtype0

sequential_3/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_namesequential_3/dense_12/kernel

0sequential_3/dense_12/kernel/Read/ReadVariableOpReadVariableOpsequential_3/dense_12/kernel*
_output_shapes
:	 *
dtype0

sequential_3/dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namesequential_3/dense_12/bias

.sequential_3/dense_12/bias/Read/ReadVariableOpReadVariableOpsequential_3/dense_12/bias*
_output_shapes	
:*
dtype0

sequential_3/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_namesequential_3/dense_13/kernel

0sequential_3/dense_13/kernel/Read/ReadVariableOpReadVariableOpsequential_3/dense_13/kernel*
_output_shapes
:	 *
dtype0

sequential_3/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namesequential_3/dense_13/bias

.sequential_3/dense_13/bias/Read/ReadVariableOpReadVariableOpsequential_3/dense_13/bias*
_output_shapes
: *
dtype0

sequential_3/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_namesequential_3/dense_14/kernel

0sequential_3/dense_14/kernel/Read/ReadVariableOpReadVariableOpsequential_3/dense_14/kernel*
_output_shapes

: *
dtype0

sequential_3/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namesequential_3/dense_14/bias

.sequential_3/dense_14/bias/Read/ReadVariableOpReadVariableOpsequential_3/dense_14/bias*
_output_shapes
:*
dtype0

sequential_3/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_namesequential_3/dense_15/kernel

0sequential_3/dense_15/kernel/Read/ReadVariableOpReadVariableOpsequential_3/dense_15/kernel*
_output_shapes
:	*
dtype0

sequential_3/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namesequential_3/dense_15/bias

.sequential_3/dense_15/bias/Read/ReadVariableOpReadVariableOpsequential_3/dense_15/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
хZ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* Z
valueZBZ BZ
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
і
0
1
2
	3

4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
 26
!27
"28
#29
$30
%31

&0
 
_]
VARIABLE_VALUEsequential_3/conv2d_12/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEsequential_3/conv2d_12/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE/sequential_3/batch_normalization_12/moving_mean,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE3sequential_3/batch_normalization_12/moving_variance,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE)sequential_3/batch_normalization_12/gamma,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(sequential_3/batch_normalization_12/beta,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEsequential_3/conv2d_13/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEsequential_3/conv2d_13/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE/sequential_3/batch_normalization_13/moving_mean,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE3sequential_3/batch_normalization_13/moving_variance,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE)sequential_3/batch_normalization_13/gamma-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE(sequential_3/batch_normalization_13/beta-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEsequential_3/conv2d_14/kernel-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEsequential_3/conv2d_14/bias-model_variables/13/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE/sequential_3/batch_normalization_14/moving_mean-model_variables/14/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE3sequential_3/batch_normalization_14/moving_variance-model_variables/15/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE)sequential_3/batch_normalization_14/gamma-model_variables/16/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE(sequential_3/batch_normalization_14/beta-model_variables/17/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEsequential_3/conv2d_15/kernel-model_variables/18/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEsequential_3/conv2d_15/bias-model_variables/19/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE/sequential_3/batch_normalization_15/moving_mean-model_variables/20/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE3sequential_3/batch_normalization_15/moving_variance-model_variables/21/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE)sequential_3/batch_normalization_15/gamma-model_variables/22/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE(sequential_3/batch_normalization_15/beta-model_variables/23/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEsequential_3/dense_12/kernel-model_variables/24/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEsequential_3/dense_12/bias-model_variables/25/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEsequential_3/dense_13/kernel-model_variables/26/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEsequential_3/dense_13/bias-model_variables/27/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEsequential_3/dense_14/kernel-model_variables/28/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEsequential_3/dense_14/bias-model_variables/29/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEsequential_3/dense_15/kernel-model_variables/30/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEsequential_3/dense_15/bias-model_variables/31/.ATTRIBUTES/VARIABLE_VALUE

'ref
'1

(
_q_network

)_layer_state_is_list
*_sequential_layers
+_layer_has_state
,	variables
-trainable_variables
.regularization_losses
/	keras_api
 

00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
 
і
0
1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
 18
!19
"20
#21
$22
%23
24
	25
26
27
28
29
30
31
Ж
0
1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
 18
!19
"20
#21
$22
%23
 
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
,	variables
-trainable_variables
.regularization_losses
h

kernel
bias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
R
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api

Qaxis
	
gamma
beta
moving_mean
	moving_variance
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
h

kernel
bias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
R
Z	variables
[trainable_variables
\regularization_losses
]	keras_api

^axis
	gamma
beta
moving_mean
moving_variance
_	variables
`trainable_variables
aregularization_losses
b	keras_api
h

kernel
bias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
R
g	variables
htrainable_variables
iregularization_losses
j	keras_api

kaxis
	gamma
beta
moving_mean
moving_variance
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
h

kernel
bias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
R
t	variables
utrainable_variables
vregularization_losses
w	keras_api

xaxis
	gamma
beta
moving_mean
moving_variance
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
S
}	variables
~trainable_variables
regularization_losses
	keras_api
l

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
l

 kernel
!bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
l

"kernel
#bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
l

$kernel
%bias
	variables
trainable_variables
regularization_losses
	keras_api
8
0
	1
2
3
4
5
6
7

00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
 
 
 

0
1

0
1
 
В
non_trainable_variables
layers
metrics
  layer_regularization_losses
Ёlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
 
 
 
В
Ђnon_trainable_variables
Ѓlayers
Єmetrics
 Ѕlayer_regularization_losses
Іlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
 


0
1
2
	3


0
1
 
В
Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
R	variables
Strainable_variables
Tregularization_losses

0
1

0
1
 
В
Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
 
 
 
В
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
 

0
1
2
3

0
1
 
В
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
_	variables
`trainable_variables
aregularization_losses

0
1

0
1
 
В
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
 
 
 
В
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
g	variables
htrainable_variables
iregularization_losses
 

0
1
2
3

0
1
 
В
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
l	variables
mtrainable_variables
nregularization_losses

0
1

0
1
 
В
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
 
 
 
В
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
t	variables
utrainable_variables
vregularization_losses
 

0
1
2
3

0
1
 
В
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
 
 
 
В
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
}	variables
~trainable_variables
regularization_losses

0
1

0
1
 
Е
оnon_trainable_variables
пlayers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
Е
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
	variables
trainable_variables
regularization_losses

 0
!1

 0
!1
 
Е
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
Е
эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
	variables
trainable_variables
regularization_losses

"0
#1

"0
#1
 
Е
ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
Е
їnon_trainable_variables
јlayers
љmetrics
 њlayer_regularization_losses
ћlayer_metrics
	variables
trainable_variables
regularization_losses

$0
%1

$0
%1
 
Е
ќnon_trainable_variables
§layers
ўmetrics
 џlayer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
 
 
 
 
 
 

0
	1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
action_0_discountPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

action_0_observationPlaceholder*/
_output_shapes
:џџџџџџџџџ
*
dtype0*$
shape:џџџџџџџџџ

j
action_0_rewardPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
m
action_0_step_typePlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
ѕ
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_typesequential_3/conv2d_12/kernelsequential_3/conv2d_12/bias)sequential_3/batch_normalization_12/gamma(sequential_3/batch_normalization_12/beta/sequential_3/batch_normalization_12/moving_mean3sequential_3/batch_normalization_12/moving_variancesequential_3/conv2d_13/kernelsequential_3/conv2d_13/bias)sequential_3/batch_normalization_13/gamma(sequential_3/batch_normalization_13/beta/sequential_3/batch_normalization_13/moving_mean3sequential_3/batch_normalization_13/moving_variancesequential_3/conv2d_14/kernelsequential_3/conv2d_14/bias)sequential_3/batch_normalization_14/gamma(sequential_3/batch_normalization_14/beta/sequential_3/batch_normalization_14/moving_mean3sequential_3/batch_normalization_14/moving_variancesequential_3/conv2d_15/kernelsequential_3/conv2d_15/bias)sequential_3/batch_normalization_15/gamma(sequential_3/batch_normalization_15/beta/sequential_3/batch_normalization_15/moving_mean3sequential_3/batch_normalization_15/moving_variancesequential_3/dense_12/kernelsequential_3/dense_12/biassequential_3/dense_13/kernelsequential_3/dense_13/biassequential_3/dense_14/kernelsequential_3/dense_14/biassequential_3/dense_15/kernelsequential_3/dense_15/bias*/
Tin(
&2$*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:џџџџџџџџџ*B
_read_only_resource_inputs$
" 	
 !"#*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 */
f*R(
&__inference_signature_wrapper_93377268
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 

PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 */
f*R(
&__inference_signature_wrapper_93377273
х
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 */
f*R(
&__inference_signature_wrapper_93377285
 
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 */
f*R(
&__inference_signature_wrapper_93377281
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Љ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp1sequential_3/conv2d_12/kernel/Read/ReadVariableOp/sequential_3/conv2d_12/bias/Read/ReadVariableOpCsequential_3/batch_normalization_12/moving_mean/Read/ReadVariableOpGsequential_3/batch_normalization_12/moving_variance/Read/ReadVariableOp=sequential_3/batch_normalization_12/gamma/Read/ReadVariableOp<sequential_3/batch_normalization_12/beta/Read/ReadVariableOp1sequential_3/conv2d_13/kernel/Read/ReadVariableOp/sequential_3/conv2d_13/bias/Read/ReadVariableOpCsequential_3/batch_normalization_13/moving_mean/Read/ReadVariableOpGsequential_3/batch_normalization_13/moving_variance/Read/ReadVariableOp=sequential_3/batch_normalization_13/gamma/Read/ReadVariableOp<sequential_3/batch_normalization_13/beta/Read/ReadVariableOp1sequential_3/conv2d_14/kernel/Read/ReadVariableOp/sequential_3/conv2d_14/bias/Read/ReadVariableOpCsequential_3/batch_normalization_14/moving_mean/Read/ReadVariableOpGsequential_3/batch_normalization_14/moving_variance/Read/ReadVariableOp=sequential_3/batch_normalization_14/gamma/Read/ReadVariableOp<sequential_3/batch_normalization_14/beta/Read/ReadVariableOp1sequential_3/conv2d_15/kernel/Read/ReadVariableOp/sequential_3/conv2d_15/bias/Read/ReadVariableOpCsequential_3/batch_normalization_15/moving_mean/Read/ReadVariableOpGsequential_3/batch_normalization_15/moving_variance/Read/ReadVariableOp=sequential_3/batch_normalization_15/gamma/Read/ReadVariableOp<sequential_3/batch_normalization_15/beta/Read/ReadVariableOp0sequential_3/dense_12/kernel/Read/ReadVariableOp.sequential_3/dense_12/bias/Read/ReadVariableOp0sequential_3/dense_13/kernel/Read/ReadVariableOp.sequential_3/dense_13/bias/Read/ReadVariableOp0sequential_3/dense_14/kernel/Read/ReadVariableOp.sequential_3/dense_14/bias/Read/ReadVariableOp0sequential_3/dense_15/kernel/Read/ReadVariableOp.sequential_3/dense_15/bias/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 **
f%R#
!__inference__traced_save_93378004

StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariablesequential_3/conv2d_12/kernelsequential_3/conv2d_12/bias/sequential_3/batch_normalization_12/moving_mean3sequential_3/batch_normalization_12/moving_variance)sequential_3/batch_normalization_12/gamma(sequential_3/batch_normalization_12/betasequential_3/conv2d_13/kernelsequential_3/conv2d_13/bias/sequential_3/batch_normalization_13/moving_mean3sequential_3/batch_normalization_13/moving_variance)sequential_3/batch_normalization_13/gamma(sequential_3/batch_normalization_13/betasequential_3/conv2d_14/kernelsequential_3/conv2d_14/bias/sequential_3/batch_normalization_14/moving_mean3sequential_3/batch_normalization_14/moving_variance)sequential_3/batch_normalization_14/gamma(sequential_3/batch_normalization_14/betasequential_3/conv2d_15/kernelsequential_3/conv2d_15/bias/sequential_3/batch_normalization_15/moving_mean3sequential_3/batch_normalization_15/moving_variance)sequential_3/batch_normalization_15/gamma(sequential_3/batch_normalization_15/betasequential_3/dense_12/kernelsequential_3/dense_12/biassequential_3/dense_13/kernelsequential_3/dense_13/biassequential_3/dense_14/kernelsequential_3/dense_14/biassequential_3/dense_15/kernelsequential_3/dense_15/bias*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *-
f(R&
$__inference__traced_restore_93378113ЂІ
н

&__inference_signature_wrapper_93377268
discount
observation

reward
	step_type"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	%

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@ 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23:	 

unknown_24:	

unknown_25:	 

unknown_26: 

unknown_27: 

unknown_28:

unknown_29:	

unknown_30:	
identity	ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*/
Tin(
&2$*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:џџџџџџџџџ*B
_read_only_resource_inputs$
" 	
 !"#*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *5
f0R.
,__inference_function_with_signature_84940332k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:^Z
/
_output_shapes
:џџџџџџџџџ

'
_user_specified_name0/observation:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:PL
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type
П
8
&__inference_get_initial_state_84941196

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
о
К"
*__inference_polymorphic_action_fn_84940902
	step_type

reward
discount
observationP
5sequential_3_conv2d_12_conv2d_readvariableop_resource:E
6sequential_3_conv2d_12_biasadd_readvariableop_resource:	J
;sequential_3_batch_normalization_12_readvariableop_resource:	L
=sequential_3_batch_normalization_12_readvariableop_1_resource:	[
Lsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	Q
5sequential_3_conv2d_13_conv2d_readvariableop_resource:E
6sequential_3_conv2d_13_biasadd_readvariableop_resource:	J
;sequential_3_batch_normalization_13_readvariableop_resource:	L
=sequential_3_batch_normalization_13_readvariableop_1_resource:	[
Lsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	P
5sequential_3_conv2d_14_conv2d_readvariableop_resource:@D
6sequential_3_conv2d_14_biasadd_readvariableop_resource:@I
;sequential_3_batch_normalization_14_readvariableop_resource:@K
=sequential_3_batch_normalization_14_readvariableop_1_resource:@Z
Lsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:@O
5sequential_3_conv2d_15_conv2d_readvariableop_resource:@ D
6sequential_3_conv2d_15_biasadd_readvariableop_resource: I
;sequential_3_batch_normalization_15_readvariableop_resource: K
=sequential_3_batch_normalization_15_readvariableop_1_resource: Z
Lsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_resource: \
Nsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource: G
4sequential_3_dense_12_matmul_readvariableop_resource:	 D
5sequential_3_dense_12_biasadd_readvariableop_resource:	G
4sequential_3_dense_13_matmul_readvariableop_resource:	 C
5sequential_3_dense_13_biasadd_readvariableop_resource: F
4sequential_3_dense_14_matmul_readvariableop_resource: C
5sequential_3_dense_14_biasadd_readvariableop_resource:G
4sequential_3_dense_15_matmul_readvariableop_resource:	D
5sequential_3_dense_15_biasadd_readvariableop_resource:	
identity	ЂCsequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_12/ReadVariableOpЂ4sequential_3/batch_normalization_12/ReadVariableOp_1ЂCsequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_13/ReadVariableOpЂ4sequential_3/batch_normalization_13/ReadVariableOp_1ЂCsequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_14/ReadVariableOpЂ4sequential_3/batch_normalization_14/ReadVariableOp_1ЂCsequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_15/ReadVariableOpЂ4sequential_3/batch_normalization_15/ReadVariableOp_1Ђ-sequential_3/conv2d_12/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_12/Conv2D/ReadVariableOpЂ-sequential_3/conv2d_13/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_13/Conv2D/ReadVariableOpЂ-sequential_3/conv2d_14/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_14/Conv2D/ReadVariableOpЂ-sequential_3/conv2d_15/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_15/Conv2D/ReadVariableOpЂ,sequential_3/dense_12/BiasAdd/ReadVariableOpЂ+sequential_3/dense_12/MatMul/ReadVariableOpЂ,sequential_3/dense_13/BiasAdd/ReadVariableOpЂ+sequential_3/dense_13/MatMul/ReadVariableOpЂ,sequential_3/dense_14/BiasAdd/ReadVariableOpЂ+sequential_3/dense_14/MatMul/ReadVariableOpЂ,sequential_3/dense_15/BiasAdd/ReadVariableOpЂ+sequential_3/dense_15/MatMul/ReadVariableOpЋ
,sequential_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0Э
sequential_3/conv2d_12/Conv2DConv2Dobservation4sequential_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
*
paddingSAME*
strides
Ё
-sequential_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0У
sequential_3/conv2d_12/BiasAddBiasAdd&sequential_3/conv2d_12/Conv2D:output:05sequential_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ

sequential_3/conv2d_12/ReluRelu'sequential_3/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
Ч
$sequential_3/max_pooling2d_6/MaxPoolMaxPool)sequential_3/conv2d_12/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
Ћ
2sequential_3/batch_normalization_12/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype0Џ
4sequential_3/batch_normalization_12/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
Csequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0б
Esequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_3/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3-sequential_3/max_pooling2d_6/MaxPool:output:0:sequential_3/batch_normalization_12/ReadVariableOp:value:0<sequential_3/batch_normalization_12/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Ќ
,sequential_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0њ
sequential_3/conv2d_13/Conv2DConv2D8sequential_3/batch_normalization_12/FusedBatchNormV3:y:04sequential_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Ё
-sequential_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0У
sequential_3/conv2d_13/BiasAddBiasAdd&sequential_3/conv2d_13/Conv2D:output:05sequential_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
sequential_3/conv2d_13/ReluRelu'sequential_3/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЧ
$sequential_3/max_pooling2d_7/MaxPoolMaxPool)sequential_3/conv2d_13/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
Ћ
2sequential_3/batch_normalization_13/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype0Џ
4sequential_3/batch_normalization_13/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
Csequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0б
Esequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_3/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3-sequential_3/max_pooling2d_7/MaxPool:output:0:sequential_3/batch_normalization_13/ReadVariableOp:value:0<sequential_3/batch_normalization_13/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Ћ
,sequential_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_14_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0љ
sequential_3/conv2d_14/Conv2DConv2D8sequential_3/batch_normalization_13/FusedBatchNormV3:y:04sequential_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
 
-sequential_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Т
sequential_3/conv2d_14/BiasAddBiasAdd&sequential_3/conv2d_14/Conv2D:output:05sequential_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
sequential_3/conv2d_14/ReluRelu'sequential_3/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@г
(sequential_3/average_pooling2d_6/AvgPoolAvgPool)sequential_3/conv2d_14/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
Њ
2sequential_3/batch_normalization_14/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_14_readvariableop_resource*
_output_shapes
:@*
dtype0Ў
4sequential_3/batch_normalization_14/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_14_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ь
Csequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0а
Esequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
4sequential_3/batch_normalization_14/FusedBatchNormV3FusedBatchNormV31sequential_3/average_pooling2d_6/AvgPool:output:0:sequential_3/batch_normalization_14/ReadVariableOp:value:0<sequential_3/batch_normalization_14/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( Њ
,sequential_3/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0љ
sequential_3/conv2d_15/Conv2DConv2D8sequential_3/batch_normalization_14/FusedBatchNormV3:y:04sequential_3/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
 
-sequential_3/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Т
sequential_3/conv2d_15/BiasAddBiasAdd&sequential_3/conv2d_15/Conv2D:output:05sequential_3/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 
sequential_3/conv2d_15/ReluRelu'sequential_3/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ г
(sequential_3/average_pooling2d_7/AvgPoolAvgPool)sequential_3/conv2d_15/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
Њ
2sequential_3/batch_normalization_15/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype0Ў
4sequential_3/batch_normalization_15/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype0Ь
Csequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0а
Esequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
4sequential_3/batch_normalization_15/FusedBatchNormV3FusedBatchNormV31sequential_3/average_pooling2d_7/AvgPool:output:0:sequential_3/batch_normalization_15/ReadVariableOp:value:0<sequential_3/batch_normalization_15/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( m
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    М
sequential_3/flatten_3/ReshapeReshape8sequential_3/batch_normalization_15/FusedBatchNormV3:y:0%sequential_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Ё
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0З
sequential_3/dense_12/MatMulMatMul'sequential_3/flatten_3/Reshape:output:03sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
,sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Й
sequential_3/dense_12/BiasAddBiasAdd&sequential_3/dense_12/MatMul:product:04sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ}
sequential_3/dense_12/ReluRelu&sequential_3/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
sequential_3/dropout_9/IdentityIdentity(sequential_3/dense_12/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0З
sequential_3/dense_13/MatMulMatMul(sequential_3/dropout_9/Identity:output:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0И
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ |
sequential_3/dense_13/ReluRelu&sequential_3/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
 sequential_3/dropout_10/IdentityIdentity(sequential_3/dense_13/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ  
+sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype0И
sequential_3/dense_14/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
,sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_3/dense_14/BiasAddBiasAdd&sequential_3/dense_14/MatMul:product:04sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
sequential_3/dense_14/ReluRelu&sequential_3/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_14/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Й
sequential_3/dense_15/MatMulMatMul)sequential_3/dropout_11/Identity:output:03sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Й
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџn
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЇ
Categorical_1/mode/ArgMaxArgMax&sequential_3/dense_15/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R f
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB n
Deterministic_1/sample/ShapeShape"Categorical_1/mode/ArgMax:output:0*
T0	*
_output_shapes
:^
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : t
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskj
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB l
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB Д
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:p
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:i
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB d
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:Џ
"Deterministic_1/sample/BroadcastToBroadcastTo"Categorical_1/mode/ArgMax:output:0&Deterministic_1/sample/concat:output:0*
T0	*'
_output_shapes
:џџџџџџџџџy
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:v
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ъ
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskf
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ў
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџZ
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value
B	 R
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:џџџџџџџџџQ
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџ
NoOpNoOpD^sequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_12/ReadVariableOp5^sequential_3/batch_normalization_12/ReadVariableOp_1D^sequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_13/ReadVariableOp5^sequential_3/batch_normalization_13/ReadVariableOp_1D^sequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_14/ReadVariableOp5^sequential_3/batch_normalization_14/ReadVariableOp_1D^sequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_15/ReadVariableOp5^sequential_3/batch_normalization_15/ReadVariableOp_1.^sequential_3/conv2d_12/BiasAdd/ReadVariableOp-^sequential_3/conv2d_12/Conv2D/ReadVariableOp.^sequential_3/conv2d_13/BiasAdd/ReadVariableOp-^sequential_3/conv2d_13/Conv2D/ReadVariableOp.^sequential_3/conv2d_14/BiasAdd/ReadVariableOp-^sequential_3/conv2d_14/Conv2D/ReadVariableOp.^sequential_3/conv2d_15/BiasAdd/ReadVariableOp-^sequential_3/conv2d_15/Conv2D/ReadVariableOp-^sequential_3/dense_12/BiasAdd/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp-^sequential_3/dense_14/BiasAdd/ReadVariableOp,^sequential_3/dense_14/MatMul/ReadVariableOp-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Csequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_12/ReadVariableOp2sequential_3/batch_normalization_12/ReadVariableOp2l
4sequential_3/batch_normalization_12/ReadVariableOp_14sequential_3/batch_normalization_12/ReadVariableOp_12
Csequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_13/ReadVariableOp2sequential_3/batch_normalization_13/ReadVariableOp2l
4sequential_3/batch_normalization_13/ReadVariableOp_14sequential_3/batch_normalization_13/ReadVariableOp_12
Csequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_14/ReadVariableOp2sequential_3/batch_normalization_14/ReadVariableOp2l
4sequential_3/batch_normalization_14/ReadVariableOp_14sequential_3/batch_normalization_14/ReadVariableOp_12
Csequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_15/ReadVariableOp2sequential_3/batch_normalization_15/ReadVariableOp2l
4sequential_3/batch_normalization_15/ReadVariableOp_14sequential_3/batch_normalization_15/ReadVariableOp_12^
-sequential_3/conv2d_12/BiasAdd/ReadVariableOp-sequential_3/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_12/Conv2D/ReadVariableOp,sequential_3/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_13/BiasAdd/ReadVariableOp-sequential_3/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_13/Conv2D/ReadVariableOp,sequential_3/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_14/BiasAdd/ReadVariableOp-sequential_3/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_14/Conv2D/ReadVariableOp,sequential_3/conv2d_14/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_15/BiasAdd/ReadVariableOp-sequential_3/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_15/Conv2D/ReadVariableOp,sequential_3/conv2d_15/Conv2D/ReadVariableOp2\
,sequential_3/dense_12/BiasAdd/ReadVariableOp,sequential_3/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp2\
,sequential_3/dense_14/BiasAdd/ReadVariableOp,sequential_3/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_14/MatMul/ReadVariableOp+sequential_3/dense_14/MatMul/ReadVariableOp2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:\X
/
_output_shapes
:џџџџџџџџџ

%
_user_specified_nameobservation
Х
N
2__inference_max_pooling2d_6_layer_call_fn_93377594

inputs
identityф
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *V
fQRO
M__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_93377294
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_93377370

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
о
Т"
*__inference_polymorphic_action_fn_84940265
	time_step
time_step_1
time_step_2
time_step_3P
5sequential_3_conv2d_12_conv2d_readvariableop_resource:E
6sequential_3_conv2d_12_biasadd_readvariableop_resource:	J
;sequential_3_batch_normalization_12_readvariableop_resource:	L
=sequential_3_batch_normalization_12_readvariableop_1_resource:	[
Lsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	Q
5sequential_3_conv2d_13_conv2d_readvariableop_resource:E
6sequential_3_conv2d_13_biasadd_readvariableop_resource:	J
;sequential_3_batch_normalization_13_readvariableop_resource:	L
=sequential_3_batch_normalization_13_readvariableop_1_resource:	[
Lsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	P
5sequential_3_conv2d_14_conv2d_readvariableop_resource:@D
6sequential_3_conv2d_14_biasadd_readvariableop_resource:@I
;sequential_3_batch_normalization_14_readvariableop_resource:@K
=sequential_3_batch_normalization_14_readvariableop_1_resource:@Z
Lsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:@O
5sequential_3_conv2d_15_conv2d_readvariableop_resource:@ D
6sequential_3_conv2d_15_biasadd_readvariableop_resource: I
;sequential_3_batch_normalization_15_readvariableop_resource: K
=sequential_3_batch_normalization_15_readvariableop_1_resource: Z
Lsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_resource: \
Nsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource: G
4sequential_3_dense_12_matmul_readvariableop_resource:	 D
5sequential_3_dense_12_biasadd_readvariableop_resource:	G
4sequential_3_dense_13_matmul_readvariableop_resource:	 C
5sequential_3_dense_13_biasadd_readvariableop_resource: F
4sequential_3_dense_14_matmul_readvariableop_resource: C
5sequential_3_dense_14_biasadd_readvariableop_resource:G
4sequential_3_dense_15_matmul_readvariableop_resource:	D
5sequential_3_dense_15_biasadd_readvariableop_resource:	
identity	ЂCsequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_12/ReadVariableOpЂ4sequential_3/batch_normalization_12/ReadVariableOp_1ЂCsequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_13/ReadVariableOpЂ4sequential_3/batch_normalization_13/ReadVariableOp_1ЂCsequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_14/ReadVariableOpЂ4sequential_3/batch_normalization_14/ReadVariableOp_1ЂCsequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_15/ReadVariableOpЂ4sequential_3/batch_normalization_15/ReadVariableOp_1Ђ-sequential_3/conv2d_12/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_12/Conv2D/ReadVariableOpЂ-sequential_3/conv2d_13/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_13/Conv2D/ReadVariableOpЂ-sequential_3/conv2d_14/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_14/Conv2D/ReadVariableOpЂ-sequential_3/conv2d_15/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_15/Conv2D/ReadVariableOpЂ,sequential_3/dense_12/BiasAdd/ReadVariableOpЂ+sequential_3/dense_12/MatMul/ReadVariableOpЂ,sequential_3/dense_13/BiasAdd/ReadVariableOpЂ+sequential_3/dense_13/MatMul/ReadVariableOpЂ,sequential_3/dense_14/BiasAdd/ReadVariableOpЂ+sequential_3/dense_14/MatMul/ReadVariableOpЂ,sequential_3/dense_15/BiasAdd/ReadVariableOpЂ+sequential_3/dense_15/MatMul/ReadVariableOpЋ
,sequential_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0Э
sequential_3/conv2d_12/Conv2DConv2Dtime_step_34sequential_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
*
paddingSAME*
strides
Ё
-sequential_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0У
sequential_3/conv2d_12/BiasAddBiasAdd&sequential_3/conv2d_12/Conv2D:output:05sequential_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ

sequential_3/conv2d_12/ReluRelu'sequential_3/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
Ч
$sequential_3/max_pooling2d_6/MaxPoolMaxPool)sequential_3/conv2d_12/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
Ћ
2sequential_3/batch_normalization_12/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype0Џ
4sequential_3/batch_normalization_12/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
Csequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0б
Esequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_3/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3-sequential_3/max_pooling2d_6/MaxPool:output:0:sequential_3/batch_normalization_12/ReadVariableOp:value:0<sequential_3/batch_normalization_12/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Ќ
,sequential_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0њ
sequential_3/conv2d_13/Conv2DConv2D8sequential_3/batch_normalization_12/FusedBatchNormV3:y:04sequential_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Ё
-sequential_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0У
sequential_3/conv2d_13/BiasAddBiasAdd&sequential_3/conv2d_13/Conv2D:output:05sequential_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
sequential_3/conv2d_13/ReluRelu'sequential_3/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЧ
$sequential_3/max_pooling2d_7/MaxPoolMaxPool)sequential_3/conv2d_13/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
Ћ
2sequential_3/batch_normalization_13/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype0Џ
4sequential_3/batch_normalization_13/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
Csequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0б
Esequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_3/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3-sequential_3/max_pooling2d_7/MaxPool:output:0:sequential_3/batch_normalization_13/ReadVariableOp:value:0<sequential_3/batch_normalization_13/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Ћ
,sequential_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_14_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0љ
sequential_3/conv2d_14/Conv2DConv2D8sequential_3/batch_normalization_13/FusedBatchNormV3:y:04sequential_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
 
-sequential_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Т
sequential_3/conv2d_14/BiasAddBiasAdd&sequential_3/conv2d_14/Conv2D:output:05sequential_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
sequential_3/conv2d_14/ReluRelu'sequential_3/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@г
(sequential_3/average_pooling2d_6/AvgPoolAvgPool)sequential_3/conv2d_14/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
Њ
2sequential_3/batch_normalization_14/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_14_readvariableop_resource*
_output_shapes
:@*
dtype0Ў
4sequential_3/batch_normalization_14/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_14_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ь
Csequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0а
Esequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
4sequential_3/batch_normalization_14/FusedBatchNormV3FusedBatchNormV31sequential_3/average_pooling2d_6/AvgPool:output:0:sequential_3/batch_normalization_14/ReadVariableOp:value:0<sequential_3/batch_normalization_14/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( Њ
,sequential_3/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0љ
sequential_3/conv2d_15/Conv2DConv2D8sequential_3/batch_normalization_14/FusedBatchNormV3:y:04sequential_3/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
 
-sequential_3/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Т
sequential_3/conv2d_15/BiasAddBiasAdd&sequential_3/conv2d_15/Conv2D:output:05sequential_3/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 
sequential_3/conv2d_15/ReluRelu'sequential_3/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ г
(sequential_3/average_pooling2d_7/AvgPoolAvgPool)sequential_3/conv2d_15/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
Њ
2sequential_3/batch_normalization_15/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype0Ў
4sequential_3/batch_normalization_15/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype0Ь
Csequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0а
Esequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
4sequential_3/batch_normalization_15/FusedBatchNormV3FusedBatchNormV31sequential_3/average_pooling2d_7/AvgPool:output:0:sequential_3/batch_normalization_15/ReadVariableOp:value:0<sequential_3/batch_normalization_15/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( m
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    М
sequential_3/flatten_3/ReshapeReshape8sequential_3/batch_normalization_15/FusedBatchNormV3:y:0%sequential_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Ё
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0З
sequential_3/dense_12/MatMulMatMul'sequential_3/flatten_3/Reshape:output:03sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
,sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Й
sequential_3/dense_12/BiasAddBiasAdd&sequential_3/dense_12/MatMul:product:04sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ}
sequential_3/dense_12/ReluRelu&sequential_3/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
sequential_3/dropout_9/IdentityIdentity(sequential_3/dense_12/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0З
sequential_3/dense_13/MatMulMatMul(sequential_3/dropout_9/Identity:output:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0И
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ |
sequential_3/dense_13/ReluRelu&sequential_3/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
 sequential_3/dropout_10/IdentityIdentity(sequential_3/dense_13/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ  
+sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype0И
sequential_3/dense_14/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
,sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_3/dense_14/BiasAddBiasAdd&sequential_3/dense_14/MatMul:product:04sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
sequential_3/dense_14/ReluRelu&sequential_3/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_14/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Й
sequential_3/dense_15/MatMulMatMul)sequential_3/dropout_11/Identity:output:03sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Й
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџn
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЇ
Categorical_1/mode/ArgMaxArgMax&sequential_3/dense_15/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R f
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB n
Deterministic_1/sample/ShapeShape"Categorical_1/mode/ArgMax:output:0*
T0	*
_output_shapes
:^
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : t
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskj
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB l
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB Д
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:p
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:i
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB d
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:Џ
"Deterministic_1/sample/BroadcastToBroadcastTo"Categorical_1/mode/ArgMax:output:0&Deterministic_1/sample/concat:output:0*
T0	*'
_output_shapes
:џџџџџџџџџy
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:v
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ъ
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskf
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ў
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџZ
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value
B	 R
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:џџџџџџџџџQ
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџ
NoOpNoOpD^sequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_12/ReadVariableOp5^sequential_3/batch_normalization_12/ReadVariableOp_1D^sequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_13/ReadVariableOp5^sequential_3/batch_normalization_13/ReadVariableOp_1D^sequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_14/ReadVariableOp5^sequential_3/batch_normalization_14/ReadVariableOp_1D^sequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_15/ReadVariableOp5^sequential_3/batch_normalization_15/ReadVariableOp_1.^sequential_3/conv2d_12/BiasAdd/ReadVariableOp-^sequential_3/conv2d_12/Conv2D/ReadVariableOp.^sequential_3/conv2d_13/BiasAdd/ReadVariableOp-^sequential_3/conv2d_13/Conv2D/ReadVariableOp.^sequential_3/conv2d_14/BiasAdd/ReadVariableOp-^sequential_3/conv2d_14/Conv2D/ReadVariableOp.^sequential_3/conv2d_15/BiasAdd/ReadVariableOp-^sequential_3/conv2d_15/Conv2D/ReadVariableOp-^sequential_3/dense_12/BiasAdd/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp-^sequential_3/dense_14/BiasAdd/ReadVariableOp,^sequential_3/dense_14/MatMul/ReadVariableOp-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Csequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_12/ReadVariableOp2sequential_3/batch_normalization_12/ReadVariableOp2l
4sequential_3/batch_normalization_12/ReadVariableOp_14sequential_3/batch_normalization_12/ReadVariableOp_12
Csequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_13/ReadVariableOp2sequential_3/batch_normalization_13/ReadVariableOp2l
4sequential_3/batch_normalization_13/ReadVariableOp_14sequential_3/batch_normalization_13/ReadVariableOp_12
Csequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_14/ReadVariableOp2sequential_3/batch_normalization_14/ReadVariableOp2l
4sequential_3/batch_normalization_14/ReadVariableOp_14sequential_3/batch_normalization_14/ReadVariableOp_12
Csequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_15/ReadVariableOp2sequential_3/batch_normalization_15/ReadVariableOp2l
4sequential_3/batch_normalization_15/ReadVariableOp_14sequential_3/batch_normalization_15/ReadVariableOp_12^
-sequential_3/conv2d_12/BiasAdd/ReadVariableOp-sequential_3/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_12/Conv2D/ReadVariableOp,sequential_3/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_13/BiasAdd/ReadVariableOp-sequential_3/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_13/Conv2D/ReadVariableOp,sequential_3/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_14/BiasAdd/ReadVariableOp-sequential_3/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_14/Conv2D/ReadVariableOp,sequential_3/conv2d_14/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_15/BiasAdd/ReadVariableOp-sequential_3/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_15/Conv2D/ReadVariableOp,sequential_3/conv2d_15/Conv2D/ReadVariableOp2\
,sequential_3/dense_12/BiasAdd/ReadVariableOp,sequential_3/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp2\
,sequential_3/dense_14/BiasAdd/ReadVariableOp,sequential_3/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_14/MatMul/ReadVariableOp+sequential_3/dense_14/MatMul/ReadVariableOp2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:ZV
/
_output_shapes
:џџџџџџџџџ

#
_user_specified_name	time_step
н
У
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93377805

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<А
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0К
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Э
R
6__inference_average_pooling2d_7_layer_call_fn_93377810

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Z
fURS
Q__inference_average_pooling2d_7_layer_call_and_return_conditional_losses_93377522
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_93377294

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_93377599

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
 	
д
9__inference_batch_normalization_14_layer_call_fn_93377756

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93377471
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Я

T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_93377547

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
п
Ѓ
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93377395

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
d
__inference_<lambda>_84438847!
readvariableop_resource:	 
identity	ЂReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
Ё
m
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_93377446

inputs
identityЊ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я

T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93377787

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
	
д
9__inference_batch_normalization_15_layer_call_fn_93377841

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *]
fXRV
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_93377578
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Ё
m
Q__inference_average_pooling2d_7_layer_call_and_return_conditional_losses_93377522

inputs
identityЊ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
э
Ч
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93377733

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<А
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0К
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј	
и
9__inference_batch_normalization_13_layer_call_fn_93377684

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93377395
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Э
R
6__inference_average_pooling2d_6_layer_call_fn_93377738

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Z
fURS
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_93377446
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
д
9__inference_batch_normalization_14_layer_call_fn_93377769

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *]
fXRV
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93377502
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
л
l
,__inference_function_with_signature_84940425
unknown:	 
identity	ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *&
f!R
__inference_<lambda>_84438847^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
 	
д
9__inference_batch_normalization_15_layer_call_fn_93377828

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *]
fXRV
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_93377547
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
п
Ѓ
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93377643

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я

T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_93377859

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_93377671

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
с

,__inference_function_with_signature_84940332
	step_type

reward
discount
observation"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	%

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@ 

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23:	 

unknown_24:	

unknown_25:	 

unknown_26: 

unknown_27: 

unknown_28:

unknown_29:	

unknown_30:	
identity	ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*/
Tin(
&2$*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:џџџџџџџџџ*B
_read_only_resource_inputs$
" 	
 !"#*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *3
f.R,
*__inference_polymorphic_action_fn_84940265k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:OK
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:^Z
/
_output_shapes
:џџџџџџџџџ

'
_user_specified_name0/observation
э
Ч
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93377661

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<А
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0К
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ьо
т"
*__inference_polymorphic_action_fn_84941060
time_step_step_type
time_step_reward
time_step_discount
time_step_observationP
5sequential_3_conv2d_12_conv2d_readvariableop_resource:E
6sequential_3_conv2d_12_biasadd_readvariableop_resource:	J
;sequential_3_batch_normalization_12_readvariableop_resource:	L
=sequential_3_batch_normalization_12_readvariableop_1_resource:	[
Lsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	Q
5sequential_3_conv2d_13_conv2d_readvariableop_resource:E
6sequential_3_conv2d_13_biasadd_readvariableop_resource:	J
;sequential_3_batch_normalization_13_readvariableop_resource:	L
=sequential_3_batch_normalization_13_readvariableop_1_resource:	[
Lsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	P
5sequential_3_conv2d_14_conv2d_readvariableop_resource:@D
6sequential_3_conv2d_14_biasadd_readvariableop_resource:@I
;sequential_3_batch_normalization_14_readvariableop_resource:@K
=sequential_3_batch_normalization_14_readvariableop_1_resource:@Z
Lsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:@O
5sequential_3_conv2d_15_conv2d_readvariableop_resource:@ D
6sequential_3_conv2d_15_biasadd_readvariableop_resource: I
;sequential_3_batch_normalization_15_readvariableop_resource: K
=sequential_3_batch_normalization_15_readvariableop_1_resource: Z
Lsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_resource: \
Nsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource: G
4sequential_3_dense_12_matmul_readvariableop_resource:	 D
5sequential_3_dense_12_biasadd_readvariableop_resource:	G
4sequential_3_dense_13_matmul_readvariableop_resource:	 C
5sequential_3_dense_13_biasadd_readvariableop_resource: F
4sequential_3_dense_14_matmul_readvariableop_resource: C
5sequential_3_dense_14_biasadd_readvariableop_resource:G
4sequential_3_dense_15_matmul_readvariableop_resource:	D
5sequential_3_dense_15_biasadd_readvariableop_resource:	
identity	ЂCsequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_12/ReadVariableOpЂ4sequential_3/batch_normalization_12/ReadVariableOp_1ЂCsequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_13/ReadVariableOpЂ4sequential_3/batch_normalization_13/ReadVariableOp_1ЂCsequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_14/ReadVariableOpЂ4sequential_3/batch_normalization_14/ReadVariableOp_1ЂCsequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_15/ReadVariableOpЂ4sequential_3/batch_normalization_15/ReadVariableOp_1Ђ-sequential_3/conv2d_12/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_12/Conv2D/ReadVariableOpЂ-sequential_3/conv2d_13/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_13/Conv2D/ReadVariableOpЂ-sequential_3/conv2d_14/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_14/Conv2D/ReadVariableOpЂ-sequential_3/conv2d_15/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_15/Conv2D/ReadVariableOpЂ,sequential_3/dense_12/BiasAdd/ReadVariableOpЂ+sequential_3/dense_12/MatMul/ReadVariableOpЂ,sequential_3/dense_13/BiasAdd/ReadVariableOpЂ+sequential_3/dense_13/MatMul/ReadVariableOpЂ,sequential_3/dense_14/BiasAdd/ReadVariableOpЂ+sequential_3/dense_14/MatMul/ReadVariableOpЂ,sequential_3/dense_15/BiasAdd/ReadVariableOpЂ+sequential_3/dense_15/MatMul/ReadVariableOpЋ
,sequential_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0з
sequential_3/conv2d_12/Conv2DConv2Dtime_step_observation4sequential_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
*
paddingSAME*
strides
Ё
-sequential_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0У
sequential_3/conv2d_12/BiasAddBiasAdd&sequential_3/conv2d_12/Conv2D:output:05sequential_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ

sequential_3/conv2d_12/ReluRelu'sequential_3/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
Ч
$sequential_3/max_pooling2d_6/MaxPoolMaxPool)sequential_3/conv2d_12/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
Ћ
2sequential_3/batch_normalization_12/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype0Џ
4sequential_3/batch_normalization_12/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
Csequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0б
Esequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_3/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3-sequential_3/max_pooling2d_6/MaxPool:output:0:sequential_3/batch_normalization_12/ReadVariableOp:value:0<sequential_3/batch_normalization_12/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Ќ
,sequential_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0њ
sequential_3/conv2d_13/Conv2DConv2D8sequential_3/batch_normalization_12/FusedBatchNormV3:y:04sequential_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Ё
-sequential_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0У
sequential_3/conv2d_13/BiasAddBiasAdd&sequential_3/conv2d_13/Conv2D:output:05sequential_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
sequential_3/conv2d_13/ReluRelu'sequential_3/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЧ
$sequential_3/max_pooling2d_7/MaxPoolMaxPool)sequential_3/conv2d_13/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
Ћ
2sequential_3/batch_normalization_13/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype0Џ
4sequential_3/batch_normalization_13/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
Csequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0б
Esequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_3/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3-sequential_3/max_pooling2d_7/MaxPool:output:0:sequential_3/batch_normalization_13/ReadVariableOp:value:0<sequential_3/batch_normalization_13/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Ћ
,sequential_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_14_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0љ
sequential_3/conv2d_14/Conv2DConv2D8sequential_3/batch_normalization_13/FusedBatchNormV3:y:04sequential_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
 
-sequential_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Т
sequential_3/conv2d_14/BiasAddBiasAdd&sequential_3/conv2d_14/Conv2D:output:05sequential_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
sequential_3/conv2d_14/ReluRelu'sequential_3/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@г
(sequential_3/average_pooling2d_6/AvgPoolAvgPool)sequential_3/conv2d_14/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
Њ
2sequential_3/batch_normalization_14/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_14_readvariableop_resource*
_output_shapes
:@*
dtype0Ў
4sequential_3/batch_normalization_14/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_14_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ь
Csequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0а
Esequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
4sequential_3/batch_normalization_14/FusedBatchNormV3FusedBatchNormV31sequential_3/average_pooling2d_6/AvgPool:output:0:sequential_3/batch_normalization_14/ReadVariableOp:value:0<sequential_3/batch_normalization_14/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( Њ
,sequential_3/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0љ
sequential_3/conv2d_15/Conv2DConv2D8sequential_3/batch_normalization_14/FusedBatchNormV3:y:04sequential_3/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
 
-sequential_3/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Т
sequential_3/conv2d_15/BiasAddBiasAdd&sequential_3/conv2d_15/Conv2D:output:05sequential_3/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 
sequential_3/conv2d_15/ReluRelu'sequential_3/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ г
(sequential_3/average_pooling2d_7/AvgPoolAvgPool)sequential_3/conv2d_15/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
Њ
2sequential_3/batch_normalization_15/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype0Ў
4sequential_3/batch_normalization_15/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype0Ь
Csequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0а
Esequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
4sequential_3/batch_normalization_15/FusedBatchNormV3FusedBatchNormV31sequential_3/average_pooling2d_7/AvgPool:output:0:sequential_3/batch_normalization_15/ReadVariableOp:value:0<sequential_3/batch_normalization_15/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( m
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    М
sequential_3/flatten_3/ReshapeReshape8sequential_3/batch_normalization_15/FusedBatchNormV3:y:0%sequential_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Ё
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0З
sequential_3/dense_12/MatMulMatMul'sequential_3/flatten_3/Reshape:output:03sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
,sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Й
sequential_3/dense_12/BiasAddBiasAdd&sequential_3/dense_12/MatMul:product:04sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ}
sequential_3/dense_12/ReluRelu&sequential_3/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
sequential_3/dropout_9/IdentityIdentity(sequential_3/dense_12/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0З
sequential_3/dense_13/MatMulMatMul(sequential_3/dropout_9/Identity:output:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0И
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ |
sequential_3/dense_13/ReluRelu&sequential_3/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
 sequential_3/dropout_10/IdentityIdentity(sequential_3/dense_13/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ  
+sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype0И
sequential_3/dense_14/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
,sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_3/dense_14/BiasAddBiasAdd&sequential_3/dense_14/MatMul:product:04sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
sequential_3/dense_14/ReluRelu&sequential_3/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_14/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Й
sequential_3/dense_15/MatMulMatMul)sequential_3/dropout_11/Identity:output:03sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Й
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџn
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЇ
Categorical_1/mode/ArgMaxArgMax&sequential_3/dense_15/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R f
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB n
Deterministic_1/sample/ShapeShape"Categorical_1/mode/ArgMax:output:0*
T0	*
_output_shapes
:^
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : t
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
$Deterministic_1/sample/strided_sliceStridedSlice%Deterministic_1/sample/Shape:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskj
'Deterministic_1/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB l
)Deterministic_1/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB Д
$Deterministic_1/sample/BroadcastArgsBroadcastArgs2Deterministic_1/sample/BroadcastArgs/s0_1:output:0-Deterministic_1/sample/strided_slice:output:0*
_output_shapes
:p
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:i
&Deterministic_1/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB d
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0/Deterministic_1/sample/concat/values_2:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:Џ
"Deterministic_1/sample/BroadcastToBroadcastTo"Categorical_1/mode/ArgMax:output:0&Deterministic_1/sample/concat:output:0*
T0	*'
_output_shapes
:џџџџџџџџџy
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:v
,Deterministic_1/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.Deterministic_1/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.Deterministic_1/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ъ
&Deterministic_1/sample/strided_slice_1StridedSlice'Deterministic_1/sample/Shape_1:output:05Deterministic_1/sample/strided_slice_1/stack:output:07Deterministic_1/sample/strided_slice_1/stack_1:output:07Deterministic_1/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskf
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0/Deterministic_1/sample/strided_slice_1:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ў
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0	*#
_output_shapes
:џџџџџџџџџZ
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value
B	 R
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:џџџџџџџџџQ
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ\
IdentityIdentityclip_by_value:z:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџ
NoOpNoOpD^sequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_12/ReadVariableOp5^sequential_3/batch_normalization_12/ReadVariableOp_1D^sequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_13/ReadVariableOp5^sequential_3/batch_normalization_13/ReadVariableOp_1D^sequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_14/ReadVariableOp5^sequential_3/batch_normalization_14/ReadVariableOp_1D^sequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_15/ReadVariableOp5^sequential_3/batch_normalization_15/ReadVariableOp_1.^sequential_3/conv2d_12/BiasAdd/ReadVariableOp-^sequential_3/conv2d_12/Conv2D/ReadVariableOp.^sequential_3/conv2d_13/BiasAdd/ReadVariableOp-^sequential_3/conv2d_13/Conv2D/ReadVariableOp.^sequential_3/conv2d_14/BiasAdd/ReadVariableOp-^sequential_3/conv2d_14/Conv2D/ReadVariableOp.^sequential_3/conv2d_15/BiasAdd/ReadVariableOp-^sequential_3/conv2d_15/Conv2D/ReadVariableOp-^sequential_3/dense_12/BiasAdd/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp-^sequential_3/dense_14/BiasAdd/ReadVariableOp,^sequential_3/dense_14/MatMul/ReadVariableOp-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Csequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_12/ReadVariableOp2sequential_3/batch_normalization_12/ReadVariableOp2l
4sequential_3/batch_normalization_12/ReadVariableOp_14sequential_3/batch_normalization_12/ReadVariableOp_12
Csequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_13/ReadVariableOp2sequential_3/batch_normalization_13/ReadVariableOp2l
4sequential_3/batch_normalization_13/ReadVariableOp_14sequential_3/batch_normalization_13/ReadVariableOp_12
Csequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_14/ReadVariableOp2sequential_3/batch_normalization_14/ReadVariableOp2l
4sequential_3/batch_normalization_14/ReadVariableOp_14sequential_3/batch_normalization_14/ReadVariableOp_12
Csequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_15/ReadVariableOp2sequential_3/batch_normalization_15/ReadVariableOp2l
4sequential_3/batch_normalization_15/ReadVariableOp_14sequential_3/batch_normalization_15/ReadVariableOp_12^
-sequential_3/conv2d_12/BiasAdd/ReadVariableOp-sequential_3/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_12/Conv2D/ReadVariableOp,sequential_3/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_13/BiasAdd/ReadVariableOp-sequential_3/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_13/Conv2D/ReadVariableOp,sequential_3/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_14/BiasAdd/ReadVariableOp-sequential_3/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_14/Conv2D/ReadVariableOp,sequential_3/conv2d_14/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_15/BiasAdd/ReadVariableOp-sequential_3/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_15/Conv2D/ReadVariableOp,sequential_3/conv2d_15/Conv2D/ReadVariableOp2\
,sequential_3/dense_12/BiasAdd/ReadVariableOp,sequential_3/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp2\
,sequential_3/dense_14/BiasAdd/ReadVariableOp,sequential_3/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_14/MatMul/ReadVariableOp+sequential_3/dense_14/MatMul/ReadVariableOp2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp:X T
#
_output_shapes
:џџџџџџџџџ
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:џџџџџџџџџ
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:џџџџџџџџџ
,
_user_specified_nametime_step/discount:fb
/
_output_shapes
:џџџџџџџџџ

/
_user_specified_nametime_step/observation
Ј	
и
9__inference_batch_normalization_12_layer_call_fn_93377612

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93377319
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
а
>
,__inference_function_with_signature_84940413

batch_size
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 */
f*R(
&__inference_get_initial_state_84940412*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
^

__inference_<lambda>_84438850*(
_construction_contextkEagerRuntime*
_input_shapes 
П
8
&__inference_get_initial_state_84940412

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size


$__inference__traced_restore_93378113
file_prefix#
assignvariableop_variable:	 K
0assignvariableop_1_sequential_3_conv2d_12_kernel:=
.assignvariableop_2_sequential_3_conv2d_12_bias:	Q
Bassignvariableop_3_sequential_3_batch_normalization_12_moving_mean:	U
Fassignvariableop_4_sequential_3_batch_normalization_12_moving_variance:	K
<assignvariableop_5_sequential_3_batch_normalization_12_gamma:	J
;assignvariableop_6_sequential_3_batch_normalization_12_beta:	L
0assignvariableop_7_sequential_3_conv2d_13_kernel:=
.assignvariableop_8_sequential_3_conv2d_13_bias:	Q
Bassignvariableop_9_sequential_3_batch_normalization_13_moving_mean:	V
Gassignvariableop_10_sequential_3_batch_normalization_13_moving_variance:	L
=assignvariableop_11_sequential_3_batch_normalization_13_gamma:	K
<assignvariableop_12_sequential_3_batch_normalization_13_beta:	L
1assignvariableop_13_sequential_3_conv2d_14_kernel:@=
/assignvariableop_14_sequential_3_conv2d_14_bias:@Q
Cassignvariableop_15_sequential_3_batch_normalization_14_moving_mean:@U
Gassignvariableop_16_sequential_3_batch_normalization_14_moving_variance:@K
=assignvariableop_17_sequential_3_batch_normalization_14_gamma:@J
<assignvariableop_18_sequential_3_batch_normalization_14_beta:@K
1assignvariableop_19_sequential_3_conv2d_15_kernel:@ =
/assignvariableop_20_sequential_3_conv2d_15_bias: Q
Cassignvariableop_21_sequential_3_batch_normalization_15_moving_mean: U
Gassignvariableop_22_sequential_3_batch_normalization_15_moving_variance: K
=assignvariableop_23_sequential_3_batch_normalization_15_gamma: J
<assignvariableop_24_sequential_3_batch_normalization_15_beta: C
0assignvariableop_25_sequential_3_dense_12_kernel:	 =
.assignvariableop_26_sequential_3_dense_12_bias:	C
0assignvariableop_27_sequential_3_dense_13_kernel:	 <
.assignvariableop_28_sequential_3_dense_13_bias: B
0assignvariableop_29_sequential_3_dense_14_kernel: <
.assignvariableop_30_sequential_3_dense_14_bias:C
0assignvariableop_31_sequential_3_dense_15_kernel:	=
.assignvariableop_32_sequential_3_dense_15_bias:	
identity_34ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*А
valueІBЃ"B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/20/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/21/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/22/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/23/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/24/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/25/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/26/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/27/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/28/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/29/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/30/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/31/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHД
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp0assignvariableop_1_sequential_3_conv2d_12_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp.assignvariableop_2_sequential_3_conv2d_12_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_3AssignVariableOpBassignvariableop_3_sequential_3_batch_normalization_12_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_4AssignVariableOpFassignvariableop_4_sequential_3_batch_normalization_12_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_5AssignVariableOp<assignvariableop_5_sequential_3_batch_normalization_12_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_6AssignVariableOp;assignvariableop_6_sequential_3_batch_normalization_12_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp0assignvariableop_7_sequential_3_conv2d_13_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp.assignvariableop_8_sequential_3_conv2d_13_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_9AssignVariableOpBassignvariableop_9_sequential_3_batch_normalization_13_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_10AssignVariableOpGassignvariableop_10_sequential_3_batch_normalization_13_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_11AssignVariableOp=assignvariableop_11_sequential_3_batch_normalization_13_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_12AssignVariableOp<assignvariableop_12_sequential_3_batch_normalization_13_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_13AssignVariableOp1assignvariableop_13_sequential_3_conv2d_14_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_14AssignVariableOp/assignvariableop_14_sequential_3_conv2d_14_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_15AssignVariableOpCassignvariableop_15_sequential_3_batch_normalization_14_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_16AssignVariableOpGassignvariableop_16_sequential_3_batch_normalization_14_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_17AssignVariableOp=assignvariableop_17_sequential_3_batch_normalization_14_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_18AssignVariableOp<assignvariableop_18_sequential_3_batch_normalization_14_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_19AssignVariableOp1assignvariableop_19_sequential_3_conv2d_15_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_20AssignVariableOp/assignvariableop_20_sequential_3_conv2d_15_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_21AssignVariableOpCassignvariableop_21_sequential_3_batch_normalization_15_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_22AssignVariableOpGassignvariableop_22_sequential_3_batch_normalization_15_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_23AssignVariableOp=assignvariableop_23_sequential_3_batch_normalization_15_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_24AssignVariableOp<assignvariableop_24_sequential_3_batch_normalization_15_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_25AssignVariableOp0assignvariableop_25_sequential_3_dense_12_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp.assignvariableop_26_sequential_3_dense_12_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_27AssignVariableOp0assignvariableop_27_sequential_3_dense_13_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp.assignvariableop_28_sequential_3_dense_13_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_29AssignVariableOp0assignvariableop_29_sequential_3_dense_14_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp.assignvariableop_30_sequential_3_dense_14_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_31AssignVariableOp0assignvariableop_31_sequential_3_dense_15_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp.assignvariableop_32_sequential_3_dense_15_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ѕ
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
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
э
Ч
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93377426

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<А
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0К
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЭС
р"
0__inference_polymorphic_distribution_fn_84941193
	step_type

reward
discount
observationP
5sequential_3_conv2d_12_conv2d_readvariableop_resource:E
6sequential_3_conv2d_12_biasadd_readvariableop_resource:	J
;sequential_3_batch_normalization_12_readvariableop_resource:	L
=sequential_3_batch_normalization_12_readvariableop_1_resource:	[
Lsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	Q
5sequential_3_conv2d_13_conv2d_readvariableop_resource:E
6sequential_3_conv2d_13_biasadd_readvariableop_resource:	J
;sequential_3_batch_normalization_13_readvariableop_resource:	L
=sequential_3_batch_normalization_13_readvariableop_1_resource:	[
Lsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	P
5sequential_3_conv2d_14_conv2d_readvariableop_resource:@D
6sequential_3_conv2d_14_biasadd_readvariableop_resource:@I
;sequential_3_batch_normalization_14_readvariableop_resource:@K
=sequential_3_batch_normalization_14_readvariableop_1_resource:@Z
Lsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:@O
5sequential_3_conv2d_15_conv2d_readvariableop_resource:@ D
6sequential_3_conv2d_15_biasadd_readvariableop_resource: I
;sequential_3_batch_normalization_15_readvariableop_resource: K
=sequential_3_batch_normalization_15_readvariableop_1_resource: Z
Lsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_resource: \
Nsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource: G
4sequential_3_dense_12_matmul_readvariableop_resource:	 D
5sequential_3_dense_12_biasadd_readvariableop_resource:	G
4sequential_3_dense_13_matmul_readvariableop_resource:	 C
5sequential_3_dense_13_biasadd_readvariableop_resource: F
4sequential_3_dense_14_matmul_readvariableop_resource: C
5sequential_3_dense_14_biasadd_readvariableop_resource:G
4sequential_3_dense_15_matmul_readvariableop_resource:	D
5sequential_3_dense_15_biasadd_readvariableop_resource:	
identity	

identity_1	

identity_2	ЂCsequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_12/ReadVariableOpЂ4sequential_3/batch_normalization_12/ReadVariableOp_1ЂCsequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_13/ReadVariableOpЂ4sequential_3/batch_normalization_13/ReadVariableOp_1ЂCsequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_14/ReadVariableOpЂ4sequential_3/batch_normalization_14/ReadVariableOp_1ЂCsequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpЂEsequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Ђ2sequential_3/batch_normalization_15/ReadVariableOpЂ4sequential_3/batch_normalization_15/ReadVariableOp_1Ђ-sequential_3/conv2d_12/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_12/Conv2D/ReadVariableOpЂ-sequential_3/conv2d_13/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_13/Conv2D/ReadVariableOpЂ-sequential_3/conv2d_14/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_14/Conv2D/ReadVariableOpЂ-sequential_3/conv2d_15/BiasAdd/ReadVariableOpЂ,sequential_3/conv2d_15/Conv2D/ReadVariableOpЂ,sequential_3/dense_12/BiasAdd/ReadVariableOpЂ+sequential_3/dense_12/MatMul/ReadVariableOpЂ,sequential_3/dense_13/BiasAdd/ReadVariableOpЂ+sequential_3/dense_13/MatMul/ReadVariableOpЂ,sequential_3/dense_14/BiasAdd/ReadVariableOpЂ+sequential_3/dense_14/MatMul/ReadVariableOpЂ,sequential_3/dense_15/BiasAdd/ReadVariableOpЂ+sequential_3/dense_15/MatMul/ReadVariableOpЋ
,sequential_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0Э
sequential_3/conv2d_12/Conv2DConv2Dobservation4sequential_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
*
paddingSAME*
strides
Ё
-sequential_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0У
sequential_3/conv2d_12/BiasAddBiasAdd&sequential_3/conv2d_12/Conv2D:output:05sequential_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ

sequential_3/conv2d_12/ReluRelu'sequential_3/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
Ч
$sequential_3/max_pooling2d_6/MaxPoolMaxPool)sequential_3/conv2d_12/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
Ћ
2sequential_3/batch_normalization_12/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype0Џ
4sequential_3/batch_normalization_12/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
Csequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0б
Esequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_3/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3-sequential_3/max_pooling2d_6/MaxPool:output:0:sequential_3/batch_normalization_12/ReadVariableOp:value:0<sequential_3/batch_normalization_12/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Ќ
,sequential_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0њ
sequential_3/conv2d_13/Conv2DConv2D8sequential_3/batch_normalization_12/FusedBatchNormV3:y:04sequential_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Ё
-sequential_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0У
sequential_3/conv2d_13/BiasAddBiasAdd&sequential_3/conv2d_13/Conv2D:output:05sequential_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
sequential_3/conv2d_13/ReluRelu'sequential_3/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЧ
$sequential_3/max_pooling2d_7/MaxPoolMaxPool)sequential_3/conv2d_13/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
Ћ
2sequential_3/batch_normalization_13/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype0Џ
4sequential_3/batch_normalization_13/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
Csequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0б
Esequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_3/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3-sequential_3/max_pooling2d_7/MaxPool:output:0:sequential_3/batch_normalization_13/ReadVariableOp:value:0<sequential_3/batch_normalization_13/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Ћ
,sequential_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_14_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0љ
sequential_3/conv2d_14/Conv2DConv2D8sequential_3/batch_normalization_13/FusedBatchNormV3:y:04sequential_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
 
-sequential_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Т
sequential_3/conv2d_14/BiasAddBiasAdd&sequential_3/conv2d_14/Conv2D:output:05sequential_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
sequential_3/conv2d_14/ReluRelu'sequential_3/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@г
(sequential_3/average_pooling2d_6/AvgPoolAvgPool)sequential_3/conv2d_14/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
Њ
2sequential_3/batch_normalization_14/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_14_readvariableop_resource*
_output_shapes
:@*
dtype0Ў
4sequential_3/batch_normalization_14/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_14_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ь
Csequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0а
Esequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
4sequential_3/batch_normalization_14/FusedBatchNormV3FusedBatchNormV31sequential_3/average_pooling2d_6/AvgPool:output:0:sequential_3/batch_normalization_14/ReadVariableOp:value:0<sequential_3/batch_normalization_14/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( Њ
,sequential_3/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0љ
sequential_3/conv2d_15/Conv2DConv2D8sequential_3/batch_normalization_14/FusedBatchNormV3:y:04sequential_3/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
 
-sequential_3/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Т
sequential_3/conv2d_15/BiasAddBiasAdd&sequential_3/conv2d_15/Conv2D:output:05sequential_3/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 
sequential_3/conv2d_15/ReluRelu'sequential_3/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ г
(sequential_3/average_pooling2d_7/AvgPoolAvgPool)sequential_3/conv2d_15/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
Њ
2sequential_3/batch_normalization_15/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype0Ў
4sequential_3/batch_normalization_15/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype0Ь
Csequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0а
Esequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
4sequential_3/batch_normalization_15/FusedBatchNormV3FusedBatchNormV31sequential_3/average_pooling2d_7/AvgPool:output:0:sequential_3/batch_normalization_15/ReadVariableOp:value:0<sequential_3/batch_normalization_15/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( m
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    М
sequential_3/flatten_3/ReshapeReshape8sequential_3/batch_normalization_15/FusedBatchNormV3:y:0%sequential_3/flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Ё
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0З
sequential_3/dense_12/MatMulMatMul'sequential_3/flatten_3/Reshape:output:03sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
,sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Й
sequential_3/dense_12/BiasAddBiasAdd&sequential_3/dense_12/MatMul:product:04sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ}
sequential_3/dense_12/ReluRelu&sequential_3/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
sequential_3/dropout_9/IdentityIdentity(sequential_3/dense_12/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0З
sequential_3/dense_13/MatMulMatMul(sequential_3/dropout_9/Identity:output:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0И
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ |
sequential_3/dense_13/ReluRelu&sequential_3/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
 sequential_3/dropout_10/IdentityIdentity(sequential_3/dense_13/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ  
+sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype0И
sequential_3/dense_14/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
,sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_3/dense_14/BiasAddBiasAdd&sequential_3/dense_14/MatMul:product:04sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
sequential_3/dense_14/ReluRelu&sequential_3/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_14/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Й
sequential_3/dense_15/MatMulMatMul)sequential_3/dropout_11/Identity:output:03sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Й
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџn
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЇ
Categorical_1/mode/ArgMaxArgMax&sequential_3/dense_15/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R T
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0	*
_output_shapes
: o

Identity_1Identity"Categorical_1/mode/ArgMax:output:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџ[

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0	*
_output_shapes
: 
NoOpNoOpD^sequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_12/ReadVariableOp5^sequential_3/batch_normalization_12/ReadVariableOp_1D^sequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_13/ReadVariableOp5^sequential_3/batch_normalization_13/ReadVariableOp_1D^sequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_14/ReadVariableOp5^sequential_3/batch_normalization_14/ReadVariableOp_1D^sequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_15/ReadVariableOp5^sequential_3/batch_normalization_15/ReadVariableOp_1.^sequential_3/conv2d_12/BiasAdd/ReadVariableOp-^sequential_3/conv2d_12/Conv2D/ReadVariableOp.^sequential_3/conv2d_13/BiasAdd/ReadVariableOp-^sequential_3/conv2d_13/Conv2D/ReadVariableOp.^sequential_3/conv2d_14/BiasAdd/ReadVariableOp-^sequential_3/conv2d_14/Conv2D/ReadVariableOp.^sequential_3/conv2d_15/BiasAdd/ReadVariableOp-^sequential_3/conv2d_15/Conv2D/ReadVariableOp-^sequential_3/dense_12/BiasAdd/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp-^sequential_3/dense_14/BiasAdd/ReadVariableOp,^sequential_3/dense_14/MatMul/ReadVariableOp-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Csequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_12/ReadVariableOp2sequential_3/batch_normalization_12/ReadVariableOp2l
4sequential_3/batch_normalization_12/ReadVariableOp_14sequential_3/batch_normalization_12/ReadVariableOp_12
Csequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_13/ReadVariableOp2sequential_3/batch_normalization_13/ReadVariableOp2l
4sequential_3/batch_normalization_13/ReadVariableOp_14sequential_3/batch_normalization_13/ReadVariableOp_12
Csequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_14/ReadVariableOp2sequential_3/batch_normalization_14/ReadVariableOp2l
4sequential_3/batch_normalization_14/ReadVariableOp_14sequential_3/batch_normalization_14/ReadVariableOp_12
Csequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2
Esequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_15/ReadVariableOp2sequential_3/batch_normalization_15/ReadVariableOp2l
4sequential_3/batch_normalization_15/ReadVariableOp_14sequential_3/batch_normalization_15/ReadVariableOp_12^
-sequential_3/conv2d_12/BiasAdd/ReadVariableOp-sequential_3/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_12/Conv2D/ReadVariableOp,sequential_3/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_13/BiasAdd/ReadVariableOp-sequential_3/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_13/Conv2D/ReadVariableOp,sequential_3/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_14/BiasAdd/ReadVariableOp-sequential_3/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_14/Conv2D/ReadVariableOp,sequential_3/conv2d_14/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_15/BiasAdd/ReadVariableOp-sequential_3/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_15/Conv2D/ReadVariableOp,sequential_3/conv2d_15/Conv2D/ReadVariableOp2\
,sequential_3/dense_12/BiasAdd/ReadVariableOp,sequential_3/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp2\
,sequential_3/dense_14/BiasAdd/ReadVariableOp,sequential_3/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_14/MatMul/ReadVariableOp+sequential_3/dense_14/MatMul/ReadVariableOp2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:\X
/
_output_shapes
:џџџџџџџџџ

%
_user_specified_nameobservation
н
У
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_93377578

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<А
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0К
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
дK
Л
!__inference__traced_save_93378004
file_prefix'
#savev2_variable_read_readvariableop	<
8savev2_sequential_3_conv2d_12_kernel_read_readvariableop:
6savev2_sequential_3_conv2d_12_bias_read_readvariableopN
Jsavev2_sequential_3_batch_normalization_12_moving_mean_read_readvariableopR
Nsavev2_sequential_3_batch_normalization_12_moving_variance_read_readvariableopH
Dsavev2_sequential_3_batch_normalization_12_gamma_read_readvariableopG
Csavev2_sequential_3_batch_normalization_12_beta_read_readvariableop<
8savev2_sequential_3_conv2d_13_kernel_read_readvariableop:
6savev2_sequential_3_conv2d_13_bias_read_readvariableopN
Jsavev2_sequential_3_batch_normalization_13_moving_mean_read_readvariableopR
Nsavev2_sequential_3_batch_normalization_13_moving_variance_read_readvariableopH
Dsavev2_sequential_3_batch_normalization_13_gamma_read_readvariableopG
Csavev2_sequential_3_batch_normalization_13_beta_read_readvariableop<
8savev2_sequential_3_conv2d_14_kernel_read_readvariableop:
6savev2_sequential_3_conv2d_14_bias_read_readvariableopN
Jsavev2_sequential_3_batch_normalization_14_moving_mean_read_readvariableopR
Nsavev2_sequential_3_batch_normalization_14_moving_variance_read_readvariableopH
Dsavev2_sequential_3_batch_normalization_14_gamma_read_readvariableopG
Csavev2_sequential_3_batch_normalization_14_beta_read_readvariableop<
8savev2_sequential_3_conv2d_15_kernel_read_readvariableop:
6savev2_sequential_3_conv2d_15_bias_read_readvariableopN
Jsavev2_sequential_3_batch_normalization_15_moving_mean_read_readvariableopR
Nsavev2_sequential_3_batch_normalization_15_moving_variance_read_readvariableopH
Dsavev2_sequential_3_batch_normalization_15_gamma_read_readvariableopG
Csavev2_sequential_3_batch_normalization_15_beta_read_readvariableop;
7savev2_sequential_3_dense_12_kernel_read_readvariableop9
5savev2_sequential_3_dense_12_bias_read_readvariableop;
7savev2_sequential_3_dense_13_kernel_read_readvariableop9
5savev2_sequential_3_dense_13_bias_read_readvariableop;
7savev2_sequential_3_dense_14_kernel_read_readvariableop9
5savev2_sequential_3_dense_14_bias_read_readvariableop;
7savev2_sequential_3_dense_15_kernel_read_readvariableop9
5savev2_sequential_3_dense_15_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*А
valueІBЃ"B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/20/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/21/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/22/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/23/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/24/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/25/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/26/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/27/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/28/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/29/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/30/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/31/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHБ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop8savev2_sequential_3_conv2d_12_kernel_read_readvariableop6savev2_sequential_3_conv2d_12_bias_read_readvariableopJsavev2_sequential_3_batch_normalization_12_moving_mean_read_readvariableopNsavev2_sequential_3_batch_normalization_12_moving_variance_read_readvariableopDsavev2_sequential_3_batch_normalization_12_gamma_read_readvariableopCsavev2_sequential_3_batch_normalization_12_beta_read_readvariableop8savev2_sequential_3_conv2d_13_kernel_read_readvariableop6savev2_sequential_3_conv2d_13_bias_read_readvariableopJsavev2_sequential_3_batch_normalization_13_moving_mean_read_readvariableopNsavev2_sequential_3_batch_normalization_13_moving_variance_read_readvariableopDsavev2_sequential_3_batch_normalization_13_gamma_read_readvariableopCsavev2_sequential_3_batch_normalization_13_beta_read_readvariableop8savev2_sequential_3_conv2d_14_kernel_read_readvariableop6savev2_sequential_3_conv2d_14_bias_read_readvariableopJsavev2_sequential_3_batch_normalization_14_moving_mean_read_readvariableopNsavev2_sequential_3_batch_normalization_14_moving_variance_read_readvariableopDsavev2_sequential_3_batch_normalization_14_gamma_read_readvariableopCsavev2_sequential_3_batch_normalization_14_beta_read_readvariableop8savev2_sequential_3_conv2d_15_kernel_read_readvariableop6savev2_sequential_3_conv2d_15_bias_read_readvariableopJsavev2_sequential_3_batch_normalization_15_moving_mean_read_readvariableopNsavev2_sequential_3_batch_normalization_15_moving_variance_read_readvariableopDsavev2_sequential_3_batch_normalization_15_gamma_read_readvariableopCsavev2_sequential_3_batch_normalization_15_beta_read_readvariableop7savev2_sequential_3_dense_12_kernel_read_readvariableop5savev2_sequential_3_dense_12_bias_read_readvariableop7savev2_sequential_3_dense_13_kernel_read_readvariableop5savev2_sequential_3_dense_13_bias_read_readvariableop7savev2_sequential_3_dense_14_kernel_read_readvariableop5savev2_sequential_3_dense_14_bias_read_readvariableop7savev2_sequential_3_dense_15_kernel_read_readvariableop5savev2_sequential_3_dense_15_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*Ў
_input_shapes
: : :::::::::::::@:@:@:@:@:@:@ : : : : : :	 ::	 : : ::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :-)
'
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!	

_output_shapes	
::!


_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::-)
'
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	 :!

_output_shapes	
::%!

_output_shapes
:	 : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::% !

_output_shapes
:	:!!

_output_shapes	
::"

_output_shapes
: 
І	
и
9__inference_batch_normalization_13_layer_call_fn_93377697

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *]
fXRV
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93377426
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
п
Ѓ
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93377319

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ё
m
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_93377743

inputs
identityЊ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я

T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93377471

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
э
Ч
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93377350

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<А
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0К
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
І	
и
9__inference_batch_normalization_12_layer_call_fn_93377625

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *]
fXRV
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93377350
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
а
8
&__inference_signature_wrapper_93377273

batch_size
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *5
f0R.
,__inference_function_with_signature_84940413*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
р
.
,__inference_function_with_signature_84940436№
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *&
f!R
__inference_<lambda>_84438850*(
_construction_contextkEagerRuntime*
_input_shapes 
щ
(
&__inference_signature_wrapper_93377285џ
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *5
f0R.
,__inference_function_with_signature_84940436*(
_construction_contextkEagerRuntime*
_input_shapes 
Х
N
2__inference_max_pooling2d_7_layer_call_fn_93377666

inputs
identityф
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *V
fQRO
M__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_93377370
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
н
У
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_93377877

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<А
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0К
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
п
Ѓ
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93377715

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ё
m
Q__inference_average_pooling2d_7_layer_call_and_return_conditional_losses_93377815

inputs
identityЊ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
н
У
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93377502

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<А
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0К
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
ф
f
&__inference_signature_wrapper_93377281
unknown:	 
identity	ЂStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *5
f0R.
,__inference_function_with_signature_84940425^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall"L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ч
actionМ
4

0/discount&
action_0_discount:0џџџџџџџџџ
F
0/observation5
action_0_observation:0џџџџџџџџџ

0
0/reward$
action_0_reward:0џџџџџџџџџ
6
0/step_type'
action_0_step_type:0џџџџџџџџџ6
action,
StatefulPartitionedCall:0	џџџџџџџџџtensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:Г
в

train_step
metadata
model_variables
_all_assets

signatures
action
distribution
get_initial_state
get_metadata
get_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper

0
1
2
	3

4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
 26
!27
"28
#29
$30
%31"
trackable_tuple_wrapper
'
&0"
trackable_list_wrapper
d
action
get_initial_state
get_train_step
get_metadata"
signature_map
8:62sequential_3/conv2d_12/kernel
*:(2sequential_3/conv2d_12/bias
@:> (2/sequential_3/batch_normalization_12/moving_mean
D:B (23sequential_3/batch_normalization_12/moving_variance
8:62)sequential_3/batch_normalization_12/gamma
7:52(sequential_3/batch_normalization_12/beta
9:72sequential_3/conv2d_13/kernel
*:(2sequential_3/conv2d_13/bias
@:> (2/sequential_3/batch_normalization_13/moving_mean
D:B (23sequential_3/batch_normalization_13/moving_variance
8:62)sequential_3/batch_normalization_13/gamma
7:52(sequential_3/batch_normalization_13/beta
8:6@2sequential_3/conv2d_14/kernel
):'@2sequential_3/conv2d_14/bias
?:=@ (2/sequential_3/batch_normalization_14/moving_mean
C:A@ (23sequential_3/batch_normalization_14/moving_variance
7:5@2)sequential_3/batch_normalization_14/gamma
6:4@2(sequential_3/batch_normalization_14/beta
7:5@ 2sequential_3/conv2d_15/kernel
):' 2sequential_3/conv2d_15/bias
?:=  (2/sequential_3/batch_normalization_15/moving_mean
C:A  (23sequential_3/batch_normalization_15/moving_variance
7:5 2)sequential_3/batch_normalization_15/gamma
6:4 2(sequential_3/batch_normalization_15/beta
/:-	 2sequential_3/dense_12/kernel
):'2sequential_3/dense_12/bias
/:-	 2sequential_3/dense_13/kernel
(:& 2sequential_3/dense_13/bias
.:, 2sequential_3/dense_14/kernel
(:&2sequential_3/dense_14/bias
/:-	2sequential_3/dense_15/kernel
):'2sequential_3/dense_15/bias
1
'ref
'1"
trackable_tuple_wrapper
.
(
_q_network"
_generic_user_object
я
)_layer_state_is_list
*_sequential_layers
+_layer_has_state
,	variables
-trainable_variables
.regularization_losses
/	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
Ж
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19"
trackable_list_wrapper
 "
trackable_list_wrapper

0
1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
 18
!19
"20
#21
$22
%23
24
	25
26
27
28
29
30
31"
trackable_list_wrapper
ж
0
1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
 18
!19
"20
#21
$22
%23"
trackable_list_wrapper
 "
trackable_list_wrapper
А
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
,	variables
-trainable_variables
.regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Н

kernel
bias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
Qaxis
	
gamma
beta
moving_mean
	moving_variance
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

kernel
bias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
^axis
	gamma
beta
moving_mean
moving_variance
_	variables
`trainable_variables
aregularization_losses
b	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

kernel
bias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
g	variables
htrainable_variables
iregularization_losses
j	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
kaxis
	gamma
beta
moving_mean
moving_variance
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

kernel
bias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
t	variables
utrainable_variables
vregularization_losses
w	keras_api
 __call__
+Ё&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
xaxis
	gamma
beta
moving_mean
moving_variance
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
Ђ__call__
+Ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ј
}	variables
~trainable_variables
regularization_losses
	keras_api
Є__call__
+Ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
С

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
І__call__
+Ї&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
Ј__call__
+Љ&call_and_return_all_conditional_losses"
_tf_keras_layer
С

 kernel
!bias
	variables
trainable_variables
regularization_losses
	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
Ќ__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layer
С

"kernel
#bias
	variables
trainable_variables
regularization_losses
	keras_api
Ў__call__
+Џ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"
_tf_keras_layer
С

$kernel
%bias
	variables
trainable_variables
regularization_losses
	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"
_tf_keras_layer
X
0
	1
2
3
4
5
6
7"
trackable_list_wrapper
Ж
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
non_trainable_variables
layers
metrics
  layer_regularization_losses
Ёlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ђnon_trainable_variables
Ѓlayers
Єmetrics
 Ѕlayer_regularization_losses
Іlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<

0
1
2
	3"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
_	variables
`trainable_variables
aregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
g	variables
htrainable_variables
iregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
t	variables
utrainable_variables
vregularization_losses
 __call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
Ђ__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
}	variables
~trainable_variables
regularization_losses
Є__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
оnon_trainable_variables
пlayers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
	variables
trainable_variables
regularization_losses
І__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
	variables
trainable_variables
regularization_losses
Ј__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
	variables
trainable_variables
regularization_losses
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
	variables
trainable_variables
regularization_losses
Ќ__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
	variables
trainable_variables
regularization_losses
Ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
їnon_trainable_variables
јlayers
љmetrics
 њlayer_regularization_losses
ћlayer_metrics
	variables
trainable_variables
regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ќnon_trainable_variables
§layers
ўmetrics
 џlayer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
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
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
2
*__inference_polymorphic_action_fn_84940902
*__inference_polymorphic_action_fn_84941060Б
ЊВІ
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsЂ
Ђ 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
щ2ц
0__inference_polymorphic_distribution_fn_84941193Б
ЊВІ
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsЂ
Ђ 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
&__inference_get_initial_state_84941196І
В
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ГBА
__inference_<lambda>_84438850"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ГBА
__inference_<lambda>_84438847"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
&__inference_signature_wrapper_93377268
0/discount0/observation0/reward0/step_type"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
аBЭ
&__inference_signature_wrapper_93377273
batch_size"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ТBП
&__inference_signature_wrapper_93377281"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ТBП
&__inference_signature_wrapper_93377285"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
т2пм
гВЯ
FullArgSpec.
args&#
jself
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaults
Ђ 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
т2пм
гВЯ
FullArgSpec.
args&#
jself
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaults
Ђ 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_max_pooling2d_6_layer_call_fn_93377594Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_93377599Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
А2­
9__inference_batch_normalization_12_layer_call_fn_93377612
9__inference_batch_normalization_12_layer_call_fn_93377625Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93377643
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93377661Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_max_pooling2d_7_layer_call_fn_93377666Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_93377671Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
А2­
9__inference_batch_normalization_13_layer_call_fn_93377684
9__inference_batch_normalization_13_layer_call_fn_93377697Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93377715
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93377733Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
р2н
6__inference_average_pooling2d_6_layer_call_fn_93377738Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћ2ј
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_93377743Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
А2­
9__inference_batch_normalization_14_layer_call_fn_93377756
9__inference_batch_normalization_14_layer_call_fn_93377769Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93377787
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93377805Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
р2н
6__inference_average_pooling2d_7_layer_call_fn_93377810Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћ2ј
Q__inference_average_pooling2d_7_layer_call_and_return_conditional_losses_93377815Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
А2­
9__inference_batch_normalization_15_layer_call_fn_93377828
9__inference_batch_normalization_15_layer_call_fn_93377841Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_93377859
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_93377877Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 <
__inference_<lambda>_84438847Ђ

Ђ 
Њ " 	5
__inference_<lambda>_84438850Ђ

Ђ 
Њ "Њ є
Q__inference_average_pooling2d_6_layer_call_and_return_conditional_losses_93377743RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ь
6__inference_average_pooling2d_6_layer_call_fn_93377738RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџє
Q__inference_average_pooling2d_7_layer_call_and_return_conditional_losses_93377815RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ь
6__inference_average_pooling2d_7_layer_call_fn_93377810RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџё
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93377643
	NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ё
T__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93377661
	NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Щ
9__inference_batch_normalization_12_layer_call_fn_93377612
	NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџЩ
9__inference_batch_normalization_12_layer_call_fn_93377625
	NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџё
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93377715NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ё
T__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93377733NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Щ
9__inference_batch_normalization_13_layer_call_fn_93377684NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџЩ
9__inference_batch_normalization_13_layer_call_fn_93377697NЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџя
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93377787MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 я
T__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93377805MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ч
9__inference_batch_normalization_14_layer_call_fn_93377756MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ч
9__inference_batch_normalization_14_layer_call_fn_93377769MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@я
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_93377859MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 я
T__inference_batch_normalization_15_layer_call_and_return_conditional_losses_93377877MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ч
9__inference_batch_normalization_15_layer_call_fn_93377828MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Ч
9__inference_batch_normalization_15_layer_call_fn_93377841MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ S
&__inference_get_initial_state_84941196)"Ђ
Ђ


batch_size 
Њ "Ђ №
M__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_93377599RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ш
2__inference_max_pooling2d_6_layer_call_fn_93377594RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ№
M__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_93377671RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ш
2__inference_max_pooling2d_7_layer_call_fn_93377666RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
*__inference_polymorphic_action_fn_84940902п 
	 !"#$%цЂт
кЂж
ЮВЪ
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџ<
observation-*
observationџџџџџџџџџ

Ђ 
Њ "RВO

PolicyStep&
action
actionџџџџџџџџџ	
stateЂ 
infoЂ Ж
*__inference_polymorphic_action_fn_84941060 
	 !"#$%Ђ
Ђў
іВђ
TimeStep6
	step_type)&
time_step/step_typeџџџџџџџџџ0
reward&#
time_step/rewardџџџџџџџџџ4
discount(%
time_step/discountџџџџџџџџџF
observation74
time_step/observationџџџџџџџџџ

Ђ 
Њ "RВO

PolicyStep&
action
actionџџџџџџџџџ	
stateЂ 
infoЂ 
0__inference_polymorphic_distribution_fn_84941193ж 
	 !"#$%цЂт
кЂж
ЮВЪ
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџ<
observation-*
observationџџџџџџџџџ

Ђ 
Њ "ШВФ

PolicyStep
actionНЂЙ
`
BЊ?

atol 	

locџџџџџџџџџ	

rtol 	
JЊG

allow_nan_statsp

namejDeterministic_1

validate_argsp 
Ђ
j
parameters
Ђ 
Ђ
jnameEtf_agents.policies.greedy_policy.DeterministicWithLogProb_ACTTypeSpec 
stateЂ 
infoЂ н
&__inference_signature_wrapper_93377268В 
	 !"#$%рЂм
Ђ 
дЊа
.

0/discount 

0/discountџџџџџџџџџ
@
0/observation/,
0/observationџџџџџџџџџ

*
0/reward
0/rewardџџџџџџџџџ
0
0/step_type!
0/step_typeџџџџџџџџџ"+Њ(
&
action
actionџџџџџџџџџ	a
&__inference_signature_wrapper_9337727370Ђ-
Ђ 
&Њ#
!

batch_size

batch_size "Њ Z
&__inference_signature_wrapper_933772810Ђ

Ђ 
Њ "Њ

int64
int64 	>
&__inference_signature_wrapper_93377285Ђ

Ђ 
Њ "Њ 