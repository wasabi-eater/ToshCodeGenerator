use std::fmt;
use std::collections::HashMap;
use std::rc::Rc;
use core::ops::*;
use std::marker::PhantomData;
#[derive(Clone)]
pub struct VarID{
    r: Rc<bool>
}
impl VarID{
    fn new() -> Self {
        Self{
            r: Rc::new(true)
        }
    }
}
impl PartialEq for VarID{
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(&*self.r, &*other.r)
    }
}
impl fmt::Debug for VarID{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error>{
        formatter.write_str("VarID")
    }
}
#[derive(Debug, Clone)]
pub enum ExprTree{
    Num(f64),
    Str(String),
    BinOp{
        left: Box<ExprTree>,
        right: Box<ExprTree>,
        op: &'static str
    },
    BinBoolOp{
        left: Box<ExprTree>,
        right: Box<ExprTree>,
        op: &'static str
    },
    StackVar{
        var_id: VarID,
        stack: Rc<str>
    },
    StackPush{
        var_id: VarID,
        expr: Box<ExprTree>,
        stack: Rc<str>
    },
    StackDelete{
        var_id: VarID,
        stack: Rc<str>
    },
    If{
        cond: Box<ExprTree>,
        then: Vec<ExprTree>,
        else_: Vec<ExprTree>
    },
    While{
        cond_statements: Vec<ExprTree>,
        cond_expr: Box<ExprTree>,
        cond_post_process: Vec<ExprTree>,
        body: Vec<ExprTree>
    },
    StackVarRewrite{
        var_id: VarID,
        stack: Rc<str>,
        expr: Box<ExprTree>
    },
    AllocateMemory {
        main_mem: Rc<str>,
        unused_mem: Rc<str>,
        expr: Vec<ExprTree>,
        var_id: VarID,
        stack: Rc<str>
    },
    FreeMemory {
        pointer: Box<ExprTree>,
        main_mem: Rc<str>,
        unused_mem: Rc<str>
    },
    CopyMemory {
        pointer: Box<ExprTree>,
        main_mem: Rc<str>,
    },
    AccessMemory{
        pointer: Box<ExprTree>,
        main_mem: Rc<str>,
        index: usize
    }
}
impl ExprTree{
    pub fn as_tosh_code(statements: impl Iterator<Item = ExprTree>) -> String{
        Self::create_tosh(statements, &mut HashMap::new()).join("\n")
    }
    fn create_tosh(statements: impl Iterator<Item = ExprTree>, stacks: &mut HashMap<Rc<str>, Vec<VarID>>) -> Vec<String> {
        let mut code: Vec<String> = vec![];
        for expr in statements{
            match expr {
                ExprTree::Num(n) => code.push(n.to_string()),
                ExprTree::Str(s) => code.push(format!("\"{}\"", s)),
                ExprTree::BinOp{left, op, right} => code.push(format!("({} {} {})",
                    Self::create_tosh(vec![*left].into_iter(), stacks).join(""),
                    op,
                    Self::create_tosh(vec![*right].into_iter(), stacks).join(""))),
                ExprTree::BinBoolOp{left, op, right} => code.push(format!("<{} {} {}>",
                    Self::create_tosh(vec![*left].into_iter(), stacks).join(""),
                    op,
                    Self::create_tosh(vec![*right].into_iter(), stacks).join(""))),
                ExprTree::StackVar{var_id, stack: stack_name} => {
                    let stack = stacks.get(&stack_name).unwrap();
                    code.push(format!("(item {} of {})",
                        stack.len() - stack.iter().position(|id| *id == var_id).unwrap(),
                        stack_name
                    ))
                },
                ExprTree::StackVarRewrite{var_id, stack: stack_name, expr} => {
                    let stack = stacks.get(&stack_name).unwrap();
                    code.push(format!("replace item {} of {} with {}",
                        stack.len() - stack.iter().position(|id| *id == var_id).unwrap(),
                        stack_name,
                        Self::create_tosh(vec![*expr].into_iter(), stacks).join("")
                    ))
                },
                ExprTree::StackPush{var_id, stack: stack_name, expr} => {
                    code.push(format!("insert {} at 1 of {}",
                        Self::create_tosh(vec![*expr].into_iter(), stacks).join(""),
                        stack_name
                    ));
                    let stack = match stacks.get_mut(&stack_name) {
                        Some(stack) => stack,
                        None => {
                            stacks.insert(stack_name.clone(), vec![]);
                            stacks.get_mut(&stack_name).unwrap()
                        }
                    };
                    stack.push(var_id);
                },
                ExprTree::StackDelete{var_id, stack: stack_name} => {
                    let stack = stacks.get_mut(&stack_name).unwrap();
                    let pos = stack.iter().position(|id| *id == var_id).unwrap();
                    stack.remove(pos);
                    code.push(format!("delete {} of {}",
                        stack.len() - pos + 1,
                        stack_name
                    ))
                },
                ExprTree::If{cond, then, else_} => {
                    let cond = Self::create_tosh(vec![*cond].into_iter(), stacks);
                    let mut else_stacks = stacks.clone();
                    code.push(format!("if {} = \"true\" then", cond.join("")));
                    code.append(&mut Self::create_tosh(then.into_iter(), stacks));
                    code.push(format!("else"));
                    code.append(&mut Self::create_tosh(else_.into_iter(), &mut else_stacks));
                    code.push("end".into());
                },
                ExprTree::While{cond_statements, cond_post_process, cond_expr, body} => {
                    let mut cond_statements = Self::create_tosh(cond_statements.into_iter(), stacks);
                    let cond_expr = Self::create_tosh(vec![*cond_expr].into_iter(), stacks);
                    let mut cond_post_process = Self::create_tosh(cond_post_process.into_iter(), stacks);
                    let mut body = Self::create_tosh(body.into_iter(), stacks);
                    code.append(&mut cond_statements.clone());
                    code.push(format!("repeat until {} = \"false\"", cond_expr.join("")));
                    code.append(&mut cond_post_process);
                    code.append(&mut body);
                    code.append(&mut cond_statements);
                    code.push("end".into());
                },
                ExprTree::AllocateMemory{main_mem, unused_mem, stack: stack_name, expr, var_id} => {
                    code.push(format!("if (length of {}) = 0 then", &unused_mem));
                    let mut then_stacks = stacks.clone();
                    code.push(format!("add 1 to {}", &main_mem));
                    {
                        code.push(format!("insert (length of {}) at 1 of {}", &*main_mem, &*stack_name));
                        let stack = match then_stacks.get_mut(&stack_name) {
                            Some(stack) => stack,
                            None => {
                                then_stacks.insert(stack_name.clone(), vec![]);
                                then_stacks.get_mut(&stack_name).unwrap()
                            }
                        };
                        stack.push(var_id.clone());
                    }
                    for expr_tree in expr.clone() {
                        code.push(format!("add {} to {}",
                            Self::create_tosh(vec![expr_tree].into_iter(), &mut then_stacks).join(""),
                            &main_mem));
                    }
                    code.push("else".into());
                    {
                        code.push(format!("insert (item 1 of {}) at 1 of {}", &*unused_mem, &*stack_name));
                        let stack = match stacks.get_mut(&stack_name) {
                            Some(stack) => stack,
                            None => {
                                stacks.insert(stack_name.clone(), vec![]);
                                stacks.get_mut(&stack_name).unwrap()
                            }
                        };
                        stack.push(var_id);
                    }
                    for (index, expr_tree) in expr.into_iter().enumerate() {
                        code.push(format!("replace item ((item 1 of {}) + {}) of {} with {}",
                            &unused_mem,
                            index,
                            &main_mem,
                            Self::create_tosh(vec![expr_tree].into_iter(), stacks).join("")));
                    }
                    code.push(format!("delete 1 of {}", &unused_mem));
                    code.push("end".into());
                },
                ExprTree::FreeMemory{pointer, main_mem, unused_mem} => {
                    let pointer = Self::create_tosh(vec![*pointer].into_iter(), stacks).join("");
                    code.push(format!("replace item {0} of {1} with ((item {0} of {1}) - 1)",
                        &pointer,
                        &main_mem
                    ));
                    code.push(format!("if (item {} of {}) = 0 then", &pointer, &main_mem));
                    code.push(format!("insert {} at 1 of {}", pointer, unused_mem));
                    code.push("end".into());
                },
                ExprTree::CopyMemory{pointer, main_mem} => {
                    let pointer = Self::create_tosh(vec![*pointer].into_iter(), stacks).join("");
                    code.push(format!("change item {0} of {1} with ((item {0} of {1}) + 1)",
                        &pointer,
                        &main_mem
                    ));
                },
                ExprTree::AccessMemory{pointer, main_mem, index} => {
                    let pointer = Self::create_tosh(vec![*pointer].into_iter(), stacks).join("");
                    code.push(format!("item ({} + {}) of {}", pointer, index + 1, main_mem));
                },
            }
        }
        code
    }
}
pub trait ExprObj {
    fn size() -> usize;
    fn copy_action<'a>(expr: &'a [ExprTree]) -> Box<dyn Iterator<Item = ExprTree>> where Self: Sized;
    fn destructor<'a>(expr: &'a [ExprTree]) -> Box<dyn Iterator<Item = ExprTree>> where Self: Sized;
}
impl ExprObj for () {
    fn size() -> usize {
        0
    }
    fn copy_action(_: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>>{
        Box::new(vec![].into_iter())
    }
    fn destructor(_: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>>{
        Box::new(vec![].into_iter())
    }
}
impl ExprObj for f64 {
    fn size() -> usize {
        1
    }
    fn copy_action(_: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>>{
        Box::new(vec![].into_iter())
    }
    fn destructor(_: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>>{
        Box::new(vec![].into_iter())
    }
}
impl ExprObj for String {
    fn size() -> usize {
        1
    }
    fn copy_action(_: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>>{
        Box::new(vec![].into_iter())
    }
    fn destructor(_: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>>{
        Box::new(vec![].into_iter())
    }
}
impl ExprObj for bool {
    fn size() -> usize{
        1
    }
    fn copy_action(_: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>>{
        Box::new(vec![].into_iter())
    }
    fn destructor(_: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>>{
        Box::new(vec![].into_iter())
    }
}
impl<T1: ExprObj, T2: ExprObj> ExprObj for (T1, T2) {
    fn size() -> usize {
        T1::size() + T2::size()
    }
    fn copy_action(expr: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>> {
        Box::new(
            T1::copy_action(&expr[0..T1::size()])
            .chain(T2::copy_action(&expr[T1::size()..expr.len()]))
        )
    }
    fn destructor(expr: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>> {
        Box::new(
            T1::destructor(&expr[0..T1::size()])
            .chain(T2::destructor(&expr[T1::size()..expr.len()]))
        )
    }
}
impl<T: ExprObj> ExprObj for Option<T> {
    fn size() -> usize{
        T::size() + 1
    }
    fn copy_action(expr: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>> {
        Box::new(
            vec![ExprTree::If{
                cond: ExprTree::BinBoolOp{
                    left: expr[0].clone().into(),
                    op: "=",
                    right: ExprTree::Str("some".into()).into()
                }.into(),
                then: T::copy_action(&expr[1..expr.len()]).collect(),
                else_: vec![]
            }].into_iter()
        )
    }
    fn destructor(expr: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>> {
        Box::new(
            vec![ExprTree::If{
                cond: ExprTree::BinBoolOp{
                    left: expr[0].clone().into(),
                    op: "=",
                    right: ExprTree::Str("some".into()).into()
                }.into(),
                then: T::destructor(&expr[1..expr.len()]).collect(),
                else_: vec![]
            }].into_iter()
        )
    }
}
impl<T: ExprObj, E: ExprObj> ExprObj for Result<T, E> {
    fn size() -> usize{
        1 + core::cmp::max(T::size(), E::size())
    }
    fn copy_action(expr: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>> {
        Box::new(
            vec![ExprTree::If{
                cond: ExprTree::BinBoolOp{
                    left: expr[0].clone().into(),
                    op: "=",
                    right: ExprTree::Str("ok".into()).into()
                }.into(),
                then: T::copy_action(&expr[1..T::size()+1]).collect(),
                else_: E::copy_action(&expr[T::size()+1..expr.len()]).collect()
            }].into_iter()
        )
    }
    fn destructor(expr: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>> {
        Box::new(
            vec![ExprTree::If{
                cond: ExprTree::BinBoolOp{
                    left: expr[0].clone().into(),
                    op: "=",
                    right: ExprTree::Str("ok".into()).into()
                }.into(),
                then: T::destructor(&expr[1..T::size()+1]).collect(),
                else_: E::destructor(&expr[T::size()+1..expr.len()]).collect()
            }].into_iter()
        )
    }
}

pub struct Expr<T : ExprObj, S: Stack>{
    statements: Box<dyn Iterator<Item = ExprTree>>,
    post_process: Box<dyn Iterator<Item = ExprTree>>,
    expr: Box<dyn Iterator<Item = ExprTree>>,
    phantom: PhantomData<(T, S)>
}
impl<S: Stack> Expr<(), S> {
    fn create_tosh(self) -> String {
        ExprTree::as_tosh_code(self.statements.chain(self.post_process))
    }
}
impl<T: ExprObj, S: Stack> Expr<T, S>{
    pub fn var<T2: ExprObj>(self, f: impl FnOnce(Variable<T, S>) -> Expr<T2, S>) -> Expr<T2, S> {
        let var_id: Vec<VarID> = (0..T::size()).map(|_| VarID::new()).collect();
        let ret = f(Variable{phantom: PhantomData, var_id: var_id.clone()});
        let statements = self.statements
            .chain(self.expr.zip(var_id.clone()).map(|(expr, var_id)|
                ExprTree::StackPush{
                    stack: S::name(),
                    var_id,
                    expr: expr.into()
                }))
            .chain(self.post_process)
            .chain(ret.statements)
            .chain(T::destructor(&*var_id.clone().into_iter().map(|var_id| ExprTree::StackVar{
                    var_id,
                    stack: S::name()
                }).collect::<Vec<_>>()));
        let post_process = ret.post_process.chain(var_id.into_iter().rev().map(|var_id|
                ExprTree::StackDelete{stack: S::name(), var_id}
            ));
        Expr {
            statements: Box::new(statements),
            post_process: Box::new(post_process),
            expr: ret.expr,
            phantom: PhantomData
        }
    }
}
impl<S: Stack> From<()> for Expr<(), S> {
    fn from(_: ()) -> Self {
        Expr {
            statements: Box::new(vec![].into_iter()),
            post_process: Box::new(vec![].into_iter()),
            expr: Box::new(vec![].into_iter()),
            phantom: PhantomData
        }
    }
}
impl <T1: Into<Expr<A, S>>, T2: Into<Expr<B, S>>, A: ExprObj, B: ExprObj, S: Stack> From<(T1, T2)> for Expr<(A, B), S> {
    fn from(tuple: (T1, T2)) -> Self {
        let (x, y) = (tuple.0.into(), tuple.1.into());
        Expr {
            statements: Box::new(x.statements.chain(y.statements)),
            post_process: Box::new(x.post_process.chain(y.post_process)),
            expr: Box::new(x.expr.chain(y.expr)),
            phantom: PhantomData
        }
    }
}
impl <T1: ExprObj, T2: ExprObj, S: Stack> Expr<(T1, T2), S> {
    pub fn item0(self) -> Expr<T1, S> {
        Expr{
            statements: self.statements,
            expr: Box::new(self.expr.take(T1::size())),
            post_process: self.post_process,
            phantom: PhantomData
        }
    }
    pub fn item1(self) -> Expr<T2, S> {
        Expr{
            statements: self.statements,
            expr: Box::new(self.expr.skip(T1::size())),
            post_process: self.post_process,
            phantom: PhantomData
        }
    }
}
impl<S: Stack> From<f64> for Expr<f64, S> {
    fn from(n: f64) -> Self {
        Expr{
            statements: Box::new(vec![].into_iter()),
            post_process: Box::new(vec![].into_iter()),
            expr: Box::new(vec![ExprTree::Num(n)].into_iter()),
            phantom: PhantomData
        }
    }
}
impl<S: Stack> Neg for Expr<f64, S>{
    type Output = Self;
    fn neg(mut self) -> Self {
        let expr = ExprTree::BinOp{
            left: ExprTree::Num(0.0).into(),
            op: "-",
            right: self.expr.nth(0).unwrap().into()
        };
        Expr{
            statements: self.statements,
            post_process: self.post_process,
            expr: Box::new(vec![expr].into_iter()),
            phantom: PhantomData
        }
    }
}
fn make_bin_op<S: Stack>(mut left: Expr<f64, S>, op: &'static str, mut right: Expr<f64, S>) -> Expr<f64, S>{
    let expr = ExprTree::BinOp {
        left: left.expr.nth(0).unwrap().into(),
        op,
        right: right.expr.nth(0).unwrap().into()
    };
    Expr {
        statements: Box::new(left.statements.chain(right.statements)),
        post_process: Box::new(left.post_process.chain(right.post_process)),
        phantom: PhantomData,
        expr: Box::new(vec![expr].into_iter())
    }
}
fn make_bin_bool_op<S: Stack>(mut left: Expr<f64, S>, op: &'static str, mut right: Expr<f64, S>) -> Expr<bool, S>{
    let expr = ExprTree::BinBoolOp {
        left: left.expr.nth(0).unwrap().into(),
        op,
        right: right.expr.nth(0).unwrap().into()
    };
    Expr {
        statements: Box::new(left.statements.chain(right.statements)),
        post_process: Box::new(left.post_process.chain(right.post_process)),
        phantom: PhantomData,
        expr: Box::new(vec![expr].into_iter())
    }
}
impl<S: Stack> Add for Expr<f64, S> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        make_bin_op(self, "+", other)
    }
}
impl<S: Stack> Sub for Expr<f64, S> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        make_bin_op(self, "-", other)
    }
}
impl<S: Stack> Mul for Expr<f64, S> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        make_bin_op(self, "*", other)
    }
}
impl<S: Stack> Div for Expr<f64, S> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        make_bin_op(self, "/", other)
    }
}
impl<S: Stack> Expr<f64, S> {
    pub fn equals(self, other: Self) -> Expr<bool, S> {
        make_bin_bool_op(self, "=", other)
    }
    pub fn less_than(self, other: Self) -> Expr<bool, S> {
        make_bin_bool_op(self, "<", other)
    }
}
impl<'a, S: Stack> From<&'a str> for Expr<String, S> {
    fn from(s: &str) -> Self {
        Expr {
            statements: Box::new(vec![].into_iter()),
            post_process: Box::new(vec![].into_iter()),
            expr: Box::new(vec![ExprTree::Str(s.into())].into_iter()),
            phantom: PhantomData
        }
    }
}
impl<S: Stack> From<bool> for Expr<bool, S> {
    fn from(b: bool) -> Self {
        Expr {
            statements: Box::new(vec![].into_iter()),
            post_process: Box::new(vec![].into_iter()),
            expr: Box::new(vec![ExprTree::Str(if b {"true"} else {"false"}.to_string())].into_iter()),
            phantom: PhantomData
        }
    }
}

pub trait Stack{
    fn name() -> Rc<str>;
}

pub struct Variable<T : ExprObj, S: Stack>{
    phantom: PhantomData<(T, S)>,
    var_id: Vec<VarID>
}
impl<T : ExprObj, S: Stack> Variable<T, S> {
    pub fn get(&self) -> Expr<T, S>{
        let expr: Vec<_> = self.var_id.iter().map(|var_id|
                ExprTree::StackVar{ stack: S::name(), var_id: var_id.clone() }
            ).collect();
        Expr{
            statements: T::copy_action(&*expr),
            post_process: Box::new(vec![].into_iter()),
            expr: Box::new(expr.into_iter()),
            phantom: PhantomData
        }
    }
}
impl<T: ExprObj, S: Stack> Expr<T, S> {
    pub fn if_(mut cond: Expr<bool, S>, then: Self, else_: Self) -> Self {
        let var_id: Vec<VarID> = (0..T::size()).map(|_| VarID::new()).collect();
        
        let then_statements = then.statements
            .chain(then.expr.zip(var_id.clone()).map(|(expr, var_id)|
                    ExprTree::StackPush{var_id, stack: S::name(), expr: expr.into()}
                ))
            .chain(then.post_process);

        let else_statements = else_.statements
            .chain(else_.expr.zip(var_id.clone()).map(|(expr, var_id)|
                    ExprTree::StackPush{var_id, stack: S::name(), expr: expr.into()}
                ))
            .chain(else_.post_process);
        
        let statements = cond.statements
            .chain(vec![ExprTree::If{
                    cond: cond.expr.nth(0).unwrap().into(),
                    then: then_statements.collect(),
                    else_: else_statements.collect()
                }]);
        
        let post_process = cond.post_process
            .chain(var_id.clone().into_iter().rev().map(|var_id|
                    ExprTree::StackDelete{stack: S::name(), var_id}
                ));
        
        let expr = var_id
            .into_iter()
            .map(|var_id| ExprTree::StackVar{var_id, stack: S::name()});
        
        Expr {
            statements: Box::new(statements),
            expr: Box::new(expr),
            post_process: Box::new(post_process),
            phantom: PhantomData
        }
    }
}

pub struct TupleExpr<T0: ExprObj, T1: ExprObj, S: Stack>{
    tuple: Expr<(T0, T1), S>
}
impl<T0: ExprObj, T1: ExprObj, S: Stack> Expr<(T0, T1), S> {
    pub fn tuple(self) -> TupleExpr<T0, T1, S>{
        TupleExpr{tuple: self}
    }
}
impl<T0: ExprObj, T1: ExprObj, S: Stack> TupleExpr<T0, T1, S> {
    pub fn var<T2: ExprObj>(self, f: impl FnOnce((Expr<T0, S>, Expr<T1, S>)) -> Expr<T2, S>) -> Expr<T2, S> {
        let mut expr: Vec<_> = self.tuple.expr.collect();
        let item1 = expr.split_off(T1::size());
        let item0 = expr;
        let ret = f((
            Expr{
                statements: Box::new(vec![].into_iter()),
                post_process: Box::new(vec![].into_iter()),
                expr: Box::new(item0.into_iter()),
                phantom: PhantomData
            },
            Expr{
                statements: Box::new(vec![].into_iter()),
                post_process: Box::new(vec![].into_iter()),
                expr: Box::new(item1.into_iter()),
                phantom: PhantomData
            }));

        Expr {
            statements: Box::new(self.tuple.statements.chain(ret.statements)),
            post_process: Box::new(ret.post_process.chain(self.tuple.post_process)),
            expr: ret.expr,
            phantom: PhantomData
        }
    }
}
impl<T: Into<Expr<A, S>>, A: ExprObj, S: Stack> From<Option<T>> for Expr<Option<A>, S> {
    fn from(op: Option<T>) -> Self{
        match op {
            Some(t) => {
                let a = t.into();
                let expr = vec![ExprTree::Str("some".to_string())].into_iter().chain(a.expr);
                Expr{
                    statements: a.statements,
                    post_process: a.post_process,
                    expr: Box::new(expr),
                    phantom: PhantomData
                }
            },
            None => {
                let expr = vec![ExprTree::Str("none".to_string())].into_iter()
                    .chain((0..A::size()).map(|_| ExprTree::Str("".to_string())));
                Expr {
                    statements:  Box::new(vec![].into_iter()),
                    post_process:  Box::new(vec![].into_iter()),
                    expr: Box::new(expr),
                    phantom: PhantomData
                }
            }
        }
    }
}
impl<T: ExprObj, S: Stack> Expr<Option<T>, S> {
    pub fn match_<T2: ExprObj>(self, some: impl FnOnce(Expr<T, S>) -> Expr<T2, S>, none: Expr<T2, S>) -> Expr<T2, S> {
        let mut expr: Vec<_> = self.expr.collect();
        let value = expr.split_off(1);
        let cond: Expr<bool, S> = Expr{
            statements: self.statements,
            expr: Box::new(vec![ExprTree::BinBoolOp{
                left: expr.pop().unwrap().into(),
                op: "=",
                right: ExprTree::Str("some".to_string()).into()
            }].into_iter()),
            post_process: self.post_process,
            phantom: PhantomData
        };
        let value: Expr<T, S> = Expr {
            statements: Box::new(vec![].into_iter()),
            expr: Box::new(value.into_iter()),
            post_process:  Box::new(vec![].into_iter()),
            phantom: PhantomData
        };
        Expr::if_(cond, some(value), none)
    }
}
impl<T: Into<Expr<T2, S>>, E: Into<Expr<E2, S>>, T2: ExprObj, E2: ExprObj, S: Stack>
    From<Result<T, E>> for Expr<Result<T2, E2>, S> {
    fn from(result: Result<T, E>) -> Self {
        let len = Result::<T2, E2>::size();
        match result {
            Ok(t) => {
                let t2 = t.into();
                let expr = vec![ExprTree::Str("ok".to_string())].into_iter()
                    .chain(t2.expr)
                    .chain((0 .. len - (T2::size() + 1)).map(|_| ExprTree::Str("".to_string())));
                Expr{
                    statements: t2.statements,
                    post_process: t2.post_process,
                    expr: Box::new(expr),
                    phantom: PhantomData
                }
            },
            Err(e) => {
                let e2 = e.into();
                let expr = vec![ExprTree::Str("error".to_string())].into_iter()
                    .chain(e2.expr)
                    .chain((0 .. len - (E2::size() + 1)).map(|_| ExprTree::Str("".to_string())));
                Expr{
                    statements: e2.statements,
                    post_process: e2.post_process,
                    expr: Box::new(expr),
                    phantom: PhantomData
                }
            }
        }
    }
}
impl<T: ExprObj, E: ExprObj, S: Stack> Expr<Result<T, E>, S> {
    pub fn match_<T2: ExprObj>(
        self,
        ok: impl FnOnce(Expr<T, S>) -> Expr<T2, S>,
        err: impl FnOnce(Expr<E, S>) -> Expr<T2, S>) -> Expr<T2, S> {
        let mut expr: Vec<_> = self.expr.collect();
        let value = expr.split_off(1);
        let cond: Expr<bool, S> = Expr{
            statements: self.statements,
            expr: Box::new(vec![ExprTree::BinBoolOp{
                left: expr.pop().unwrap().into(),
                op: "=",
                right: ExprTree::Str("ok".to_string()).into()
            }].into_iter()),
            post_process: self.post_process,
            phantom: PhantomData
        };
        let ok_value: Expr<T, S> = Expr {
            statements: Box::new(vec![].into_iter()),
            expr: Box::new(value.clone().into_iter().take(T::size())),
            post_process: Box::new(vec![].into_iter()),
            phantom: PhantomData
        };
        let err_value: Expr<E, S> = Expr {
            statements: Box::new(vec![].into_iter()),
            expr: Box::new(value.into_iter().take(E::size())),
            post_process: Box::new(vec![].into_iter()),
            phantom: PhantomData
        };
        Expr::if_(cond, ok(ok_value), err(err_value))
    }
}
pub struct Mut<T: ExprObj, S: Stack>{
    expr: Expr<T, S>
}
impl<T: ExprObj, S: Stack> Expr<T, S>{
    pub fn mut_(self) -> Mut<T, S> {
        Mut{expr: self}
    }
}
pub struct MutVariable<T: ExprObj, S: Stack>{
    phantom: PhantomData<(T, S)>,
    var_id: Vec<VarID>
}
impl<T: ExprObj, S: Stack> MutVariable<T, S> {
    pub fn get(&self) -> Expr<T, S>{
        let expr: Vec<_> = self.var_id.iter().map(|var_id|
                    ExprTree::StackVar{ stack: S::name(), var_id: var_id.clone() }
                ).collect();
        Expr{
            statements: T::copy_action(&expr),
            post_process: Box::new(vec![].into_iter()),
            expr: Box::new(expr.into_iter()),
            phantom: PhantomData
        }
    }
    pub fn rewrite(&self, expr: Expr<T, S>) -> Expr<(), S> {
        let statements =
            expr.expr.into_iter().zip(self.var_id.clone().into_iter()).map(|(expr, var_id)|
                    ExprTree::StackVarRewrite{ expr: expr.into(), stack: S::name(), var_id }
                )
            .chain(expr.post_process.into_iter());

        Expr{
            statements: Box::new(statements),
            post_process: Box::new(vec![].into_iter()),
            expr: Box::new(vec![].into_iter()),
            phantom: PhantomData
        }
    }
}
impl<T: ExprObj, S: Stack> Mut<T, S> {
    pub fn var<T2: ExprObj>(self, f: impl FnOnce(MutVariable<T, S>) -> Expr<T2, S>) -> Expr<T2, S> {
        let var_id: Vec<VarID> = (0..T::size()).map(|_| VarID::new()).collect();

        let ret = f(MutVariable{phantom: PhantomData, var_id: var_id.clone()});

        let statements = self.expr.statements
            .chain(self.expr.expr.zip(var_id.clone()).map(|(expr, var_id)|
                ExprTree::StackPush{var_id, stack: S::name(), expr: expr.into()}
            ))
            .chain(self.expr.post_process)
            .chain(ret.statements)
            .chain(T::destructor(&*var_id.clone().into_iter().map(|var_id|
                    ExprTree::StackVar{ var_id, stack: S::name() }).collect::<Vec<_>>()
                ));

        let post_process = ret.post_process
                .chain(var_id.into_iter().rev().map(|var_id|
                        ExprTree::StackDelete{stack: S::name(), var_id}
                    ));

        Expr {
            statements: Box::new(statements),
            post_process: Box::new(post_process),
            expr: ret.expr,
            phantom: PhantomData
        }
    }
}

impl<S: Stack> Expr<(), S> {
    pub fn while_(mut cond: Expr<bool, S>, body: Expr<(), S>) -> Expr<(), S>{
        Expr {
            statements: Box::new(vec![ExprTree::While{
                cond_statements: cond.statements.collect(),
                cond_expr: cond.expr.nth(0).unwrap().into(),
                cond_post_process: cond.post_process.collect(),
                body: body.statements.chain(body.post_process).collect()
            }].into_iter()),
            post_process: Box::new(vec![].into_iter()),
            expr: Box::new(vec![].into_iter()),
            phantom: PhantomData
        }
    }
}

pub trait HeapMemory<T: ExprObj>{
    fn main_mem() -> Rc<str>;
    fn unused_mem() -> Rc<str>;
}

pub struct Allocater<T: ExprObj, H: HeapMemory<T>, S: Stack> {
    expr: Expr<T, S>,
    phantom: PhantomData<Box<H>>
}
impl<T: ExprObj, H: HeapMemory<T>, S: Stack> Allocater<T, H, S> {
    pub fn alloc(_: H, expr: Expr<T, S>) -> Self {
        Allocater{expr, phantom: PhantomData}
    }
    pub fn var<T2: ExprObj>(self, f: impl FnOnce(Expr<Mem<T, H>, S>) -> Expr<T2, S>) -> Expr<T2, S> {
        let var_id = VarID::new();
        
        let r = f(Expr {
            statements: Box::new(vec![].into_iter()),
            expr: Box::new(vec![ExprTree::StackVar{var_id: var_id.clone(), stack: S::name()}].into_iter()),
            post_process: Box::new(vec![].into_iter()),
            phantom: PhantomData
        });
        
        let statements = self.expr.statements
            .chain(vec![ExprTree::AllocateMemory{
                    main_mem: H::main_mem(),
                    unused_mem: H::unused_mem(),
                    expr: self.expr.expr.collect(),
                    var_id,
                    stack: S::name()
                }])
            .chain(r.statements);

        let post_process = r.post_process.chain(self.expr.post_process);

        Expr{
            statements: Box::new(statements),
            post_process: Box::new(post_process),
            expr: r.expr,
            phantom: PhantomData
        }
    }
}

pub struct Mem<T: ExprObj, H: HeapMemory<T>> {
    phantom: PhantomData<(T, Box<H>)>
}
impl<T: ExprObj, H: HeapMemory<T>> ExprObj for Mem<T, H> {
    fn size() -> usize {
        1
    }
    fn copy_action(expr: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>>{
        Box::new(
            vec![
                ExprTree::CopyMemory{
                    pointer: expr[0].clone().into(),
                    main_mem: H::main_mem()
                }
            ].into_iter()
        )
    }
    fn destructor(expr: &[ExprTree]) -> Box<dyn Iterator<Item = ExprTree>>{
        Box::new(
            vec![
                ExprTree::FreeMemory{
                    pointer: expr[0].clone().into(),
                    main_mem: H::main_mem(),
                    unused_mem: H::unused_mem()
                }
            ].into_iter()
        )
    }
}

impl <T: ExprObj, H: HeapMemory<T>, S: Stack> Expr<Mem<T, H>, S> {
    pub fn get(mut self) -> Expr<T, S> {
        let pointer = self.expr.nth(0).unwrap();
        
        let post_process = Mem::<T, H>::destructor(&[pointer.clone()])
            .chain(self.post_process);
        
        let expr = (0..T::size()).map(move |index| ExprTree::AccessMemory{
                pointer: pointer.clone().into(),
                index,
                main_mem: H::main_mem()
            });
        
        Expr {
            statements: self.statements,
            expr: Box::new(expr),
            post_process: Box::new(post_process),
            phantom: PhantomData
        }
    }
}

macro_rules! action {
    {let! $v:pat = $e:expr; $(let! $v2:pat = $e2:expr;)* $e3:expr} => {
        $e.var(|$v| action!{$(let! $v2 = $e2;)* $e3})
    };
    {$e:expr} => ($e);
}

fn main(){
    #[derive(Debug)]
    struct ST;
    impl Stack for ST{
        fn name() -> Rc<str> {
            Rc::from("stack")
        }
    }
    let expr: Expr<(), ST> = action!{
        let! x = Expr::from(-5.0);
        
        let! (x, y) = Expr::if_(x.get().less_than(Expr::from(0.0)),
            Expr::from((-x.get(), -1.0)),
            Expr::from((x.get(), 1.0))
        ).tuple();
        
        Expr::from(())
    };
    println!("{}", expr.create_tosh());
}
