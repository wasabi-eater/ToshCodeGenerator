use std::fmt;
use std::collections::HashMap;
use std::rc::Rc;
use core::ops::*;
use std::marker::PhantomData;
#[derive(Clone)]
struct VarID{
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
enum ExprTree{
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
        heap_mem: Rc<str>,
        unused_mem: Rc<str>,
        stack: Rc<str>
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
                ExprTree::AllocateMemory{heap_mem, unused_mem, stack} => todo!()
            }
        }
        code
    }
}
pub trait FieldSized {
    fn size() -> usize;
}
impl FieldSized for () {
    fn size() -> usize {
        0
    }
}
impl FieldSized for f64 {
    fn size() -> usize {
        1
    }
}
impl FieldSized for String {
    fn size() -> usize {
        1
    }
}
impl FieldSized for bool {
    fn size() -> usize{
        1
    }
}
impl<T1: FieldSized, T2: FieldSized> FieldSized for (T1, T2) {
    fn size() -> usize {
        T1::size() + T2::size()
    }
}
impl<T: FieldSized> FieldSized for Option<T> {
    fn size() -> usize{
        T::size() + 1
    }
}
impl<T: FieldSized, E: FieldSized> FieldSized for Result<T, E> {
    fn size() -> usize{
        1 + core::cmp::max(T::size(), E::size())
    }
}
#[derive(Debug)]
pub struct Expr<T : FieldSized>{
    statements: Vec<ExprTree>,
    post_process: Vec<ExprTree>,
    expr: Vec<ExprTree>,
    phantom: PhantomData<T>
}
impl Expr<()> {
    fn create_tosh(self) -> String {
        ExprTree::as_tosh_code(self.statements.into_iter().chain(self.post_process))
    }
}
impl<T : FieldSized> Expr<T>{
    pub fn var<T2: FieldSized>(mut self, stack: &Stack, f: impl FnOnce(Variable<T>) -> Expr<T2>) -> Expr<T2> {
        let expr_count = self.expr.len();
        let var_id: Vec<VarID> = (0..expr_count).map(|_| VarID::new()).collect();
        let mut ret = f(Variable{stack: stack.name.clone(), phantom: PhantomData, var_id: var_id.clone()});
        let mut statements = self.statements;
        for (expr, var_id) in self.expr.into_iter().zip(var_id.clone()) {
            statements.push(ExprTree::StackPush{
                var_id,
                stack: stack.name.clone(),
                expr: expr.into()
            });
        }
        statements.append(&mut self.post_process);
        statements.append(&mut ret.statements);
        for var_id in var_id.into_iter().rev() {
            ret.post_process.push(ExprTree::StackDelete{stack: stack.name.clone(), var_id});
        }
        Expr {
            statements,
            post_process: ret.post_process,
            expr: ret.expr,
            phantom: PhantomData
        }
    }
}
impl From<()> for Expr<()> {
    fn from(_: ()) -> Self {
        Expr {
            statements: vec![],
            post_process: vec![],
            expr: vec![],
            phantom: PhantomData
        }
    }
}
impl <T1: Into<Expr<A>>, T2: Into<Expr<B>>, A: FieldSized, B: FieldSized> From<(T1, T2)> for Expr<(A, B)> {
    fn from(tuple: (T1, T2)) -> Self {
        let (mut x, mut y) = (tuple.0.into(), tuple.1.into());
        let mut statements = x.statements;
        statements.append(&mut y.statements);
        let mut post_process = y.post_process;
        post_process.append(&mut x.post_process);
        let mut expr = x.expr;
        expr.append(&mut y.expr);
        Expr {
            statements,
            post_process,
            expr,
            phantom: PhantomData
        }
    }
}
impl <T1: FieldSized, T2: FieldSized> Expr<(T1, T2)> {
    pub fn item0(mut self) -> Expr<T1> {
        let _ = self.expr.split_off(T1::size());
        Expr{
            statements: self.statements,
            expr: self.expr,
            post_process: self.post_process,
            phantom: PhantomData
        }
    }
    pub fn item1(mut self) -> Expr<T2> {
        let t2_expr = self.expr.split_off(T1::size());
        Expr{
            statements: self.statements,
            expr: t2_expr,
            post_process: self.post_process,
            phantom: PhantomData
        }
    }
}
impl From<f64> for Expr<f64> {
    fn from(n: f64) -> Self {
        Expr{
            statements: vec![],
            post_process: vec![],
            expr: vec![ExprTree::Num(n)],
            phantom: PhantomData
        }
    }
}
impl Neg for Expr<f64>{
    type Output = Self;
    fn neg(mut self) -> Self {
        Expr{
            statements: self.statements,
            post_process: self.post_process,
            expr: vec![ExprTree::BinOp{
                left: ExprTree::Num(0.0).into(),
                op: "-",
                right: self.expr.pop().unwrap().into()
            }],
            phantom: PhantomData
        }
    }
}
fn make_bin_op(mut left: Expr<f64>, op: &'static str, mut right: Expr<f64>) -> Expr<f64>{
    let mut statements = left.statements;
    statements.append(&mut right.statements);
    let mut post_process = left.post_process;
    post_process.append(&mut right.post_process);
    Expr {
        statements,
        post_process,
        phantom: PhantomData,
        expr: vec![
            ExprTree::BinOp {
                left: left.expr.pop().unwrap().into(),
                op,
                right: right.expr.pop().unwrap().into()
            }
        ]
    }
}
impl Add for Expr<f64> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        make_bin_op(self, "+", other)
    }
}
impl Sub for Expr<f64> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        make_bin_op(self, "-", other)
    }
}
impl Mul for Expr<f64> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        make_bin_op(self, "*", other)
    }
}
impl Div for Expr<f64> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        make_bin_op(self, "/", other)
    }
}
impl Expr<f64> {
    pub fn equals(mut self, mut other: Self) -> Expr<bool> {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        let mut post_process = self.post_process;
        post_process.append(&mut other.post_process);
        Expr {
            statements,
            post_process,
            expr: vec![ExprTree::BinBoolOp{
                left: self.expr.pop().unwrap().into(),
                op: "=",
                right: other.expr.pop().unwrap().into(),
            }],
            phantom: PhantomData
        }
    }
}
impl<'a> From<&'a str> for Expr<String> {
    fn from(s: &str) -> Self {
        Expr {
            statements: vec![],
            post_process: vec![],
            expr: vec![ExprTree::Str(s.into())],
            phantom: PhantomData
        }
    }
}
impl From<bool> for Expr<bool> {
    fn from(b: bool) -> Self {
        Expr {
            statements: vec![],
            post_process: vec![],
            expr: vec![ExprTree::Str(if b {"true"} else {"false"}.to_string())],
            phantom: PhantomData
        }
    }
}

pub struct Stack{
    name: Rc<str>
}
impl Stack{
    pub fn new(name: &str) -> Self {
        Self {name: name.into()}
    }
}

pub struct Variable<T : FieldSized>{
    stack: Rc<str>,
    phantom: PhantomData<T>,
    var_id: Vec<VarID>
}
impl<T : FieldSized> Variable<T> {
    pub fn get(&self) -> Expr<T>{
        Expr{
            statements: vec![],
            post_process: vec![],
            expr: self.var_id.iter().map(|var_id|
                    ExprTree::StackVar{
                        stack: self.stack.clone(),
                        var_id: var_id.clone()
                    }
                ).collect(),
            phantom: PhantomData
        }
    }
}
pub struct If<T: FieldSized>{
    cond: Expr<bool>,
    then: Expr<T>,
    else_: Expr<T>
}
impl<T: FieldSized> Expr<T> {
    pub fn if_(cond: Expr<bool>, then: Expr<T>, else_: Expr<T>) -> If<T> {
        If {cond, then, else_}
    }
}
impl<T: FieldSized> If<T> {
    pub fn var<T2 : FieldSized>(mut self, stack: &Stack, f: impl FnOnce(Variable<T>) -> Expr<T2>) -> Expr<T2> {
        let expr_count = T::size();
        let var_id: Vec<VarID> = (0..expr_count).map(|_| VarID::new()).collect();
        let mut ret = f(Variable{stack: stack.name.clone(), phantom: PhantomData, var_id: var_id.clone()});
        let mut then = self.then.statements;
        let mut else_ = self.else_.statements;
        for (expr, var_id) in self.then.expr.into_iter().zip(var_id.clone()) {
            then.push(ExprTree::StackPush{
                var_id,
                stack: stack.name.clone(),
                expr: expr.into()
            });
        }
        for (expr, var_id) in self.else_.expr.into_iter().zip(var_id.clone()) {
            else_.push(ExprTree::StackPush{
                var_id,
                stack: stack.name.clone(),
                expr: expr.into()
            });
        }
        then.append(&mut self.then.post_process);
        else_.append(&mut self.else_.post_process);
        let mut statements = self.cond.statements;
        statements.push(ExprTree::If{
           cond: self.cond.expr.pop().unwrap().into(),
           then,
           else_
        });
        statements.append(&mut ret.statements);
        let mut post_process = self.cond.post_process;
        post_process.append(&mut ret.post_process);
        for var_id in var_id.into_iter().rev() {
            post_process.push(ExprTree::StackDelete{stack: stack.name.clone(), var_id});
        }
        
        Expr {
            statements,
            post_process,
            expr: ret.expr,
            phantom: PhantomData
        }
    }
}
pub struct TupleExpr<T0: FieldSized, T1: FieldSized>{
    tuple: Expr<(T0, T1)>
}
impl<T0: FieldSized, T1: FieldSized> Expr<(T0, T1)> {
    pub fn tuple(self) -> TupleExpr<T0, T1>{
        TupleExpr{tuple: self}
    }
}
impl<T0: FieldSized, T1: FieldSized> TupleExpr<T0, T1> {
    pub fn var<T2: FieldSized>(
        mut self, _: &Stack,
        f: impl FnOnce((Expr<T0>, Expr<T1>)) -> Expr<T2>) -> Expr<T2> {
        let item1 = self.tuple.expr.split_off(T1::size());
        let item0 = self.tuple.expr;
        let mut ret = f((
            Expr{
                statements: vec![],
                post_process: vec![],
                expr: item0,
                phantom: PhantomData
            },
            Expr{
                statements: vec![],
                post_process: vec![],
                expr: item1,
                phantom: PhantomData
            }));
        let mut statements = self.tuple.statements;
        statements.append(&mut ret.statements);
        ret.post_process.append(&mut self.tuple.post_process);
        Expr {
            statements,
            post_process: ret.post_process,
            expr: ret.expr,
            phantom: PhantomData
        }
    }
}
impl<T: Into<Expr<A>>, A: FieldSized> From<Option<T>> for Expr<Option<A>> {
    fn from(op: Option<T>) -> Self{
        match op {
            Some(t) => {
                let mut a = t.into();
                a.expr.insert(0, ExprTree::Str("some".to_string()));
                Expr{
                    statements: a.statements,
                    post_process: a.post_process,
                    expr: a.expr,
                    phantom: PhantomData
                }
            },
            None => {
                let expr = vec![ExprTree::Str("none".to_string())].into_iter()
                    .chain([|| ExprTree::Num(0.0)].iter().cycle().take(A::size()).map(|f| f())).collect();
                Expr {
                    statements: vec![],
                    post_process: vec![],
                    expr,
                    phantom: PhantomData
                }
            }
        }
    }
}
impl<T: FieldSized> Expr<Option<T>> {
    pub fn match_<T2: FieldSized>(mut self, some: impl FnOnce(Expr<T>) -> Expr<T2>, none: Expr<T2>) -> If<T2> {
        let value = self.expr.split_off(1);
        let cond: Expr<bool> = Expr{
            statements: self.statements,
            expr: vec![ExprTree::BinBoolOp{
                left: self.expr.pop().unwrap().into(),
                op: "=",
                right: ExprTree::Str("some".to_string()).into()
            }],
            post_process: self.post_process,
            phantom: PhantomData
        };
        let value: Expr<T> = Expr {
            statements: vec![],
            expr: value,
            post_process: vec![],
            phantom: PhantomData
        };
        Expr::if_(cond, some(value), none)
    }
}
impl<T: Into<Expr<T2>>, E: Into<Expr<E2>>, T2: FieldSized, E2: FieldSized>
    From<Result<T, E>> for Expr<Result<T2, E2>> {
    fn from(result: Result<T, E>) -> Self {
        let len = Result::<T2, E2>::size();
        match result {
            Ok(t) => {
                let t2 = t.into();
                let zero = [|| ExprTree::Num(0.0)].iter().cycle().take(len - (1 + T2::size())).map(|f| f());
                Expr{
                    statements: t2.statements,
                    post_process: t2.post_process,
                    expr: vec![ExprTree::Str("ok".to_string())].into_iter().chain(t2.expr).chain(zero).collect(),
                    phantom: PhantomData
                }
            },
            Err(e) => {
                let e2 = e.into();
                let zero = [|| ExprTree::Num(0.0)].iter().cycle().take(len - (1 + E2::size())).map(|f| f());
                Expr{
                    statements: e2.statements,
                    post_process: e2.post_process,
                    expr: vec![ExprTree::Str("err".to_string())].into_iter().chain(e2.expr).chain(zero).collect(),
                    phantom: PhantomData
                }
            }
        }
    }
}
impl<T: FieldSized, E: FieldSized> Expr<Result<T, E>> {
    pub fn match_<T2: FieldSized>(
        mut self,
        ok: impl FnOnce(Expr<T>) -> Expr<T2>,
        err: impl FnOnce(Expr<E>) -> Expr<T2>) -> If<T2> {
        let value = self.expr.split_off(1);
        let cond: Expr<bool> = Expr{
            statements: self.statements,
            expr: vec![ExprTree::BinBoolOp{
                left: self.expr.pop().unwrap().into(),
                op: "=",
                right: ExprTree::Str("ok".to_string()).into()
            }],
            post_process: self.post_process,
            phantom: PhantomData
        };
        let ok_value: Expr<T> = Expr {
            statements: vec![],
            expr: value.clone().into_iter().take(T::size()).collect(),
            post_process: vec![],
            phantom: PhantomData
        };
        let err_value: Expr<E> = Expr {
            statements: vec![],
            expr: value.into_iter().take(E::size()).collect(),
            post_process: vec![],
            phantom: PhantomData
        };
        Expr::if_(cond, ok(ok_value), err(err_value))
    }
}
pub struct Mut<T: FieldSized>{
    expr: Expr<T>
}
impl<T: FieldSized> Expr<T>{
    pub fn mut_(self) -> Mut<T> {
        Mut{expr: self}
    }
}
pub struct MutVariable<T : FieldSized>{
    stack: Rc<str>,
    phantom: PhantomData<T>,
    var_id: Vec<VarID>
}
impl<T : FieldSized> MutVariable<T> {
    pub fn get(&self) -> Expr<T>{
        Expr{
            statements: vec![],
            post_process: vec![],
            expr: self.var_id.iter().map(|var_id|
                    ExprTree::StackVar{
                        stack: self.stack.clone(),
                        var_id: var_id.clone()
                    }
                ).collect(),
            phantom: PhantomData
        }
    }
    pub fn rewrite(&self, stack: &Stack, expr: Expr<T>) -> Expr<()> {
        Expr{
            statements: expr.expr.into_iter().zip(self.var_id.clone().into_iter()).map(|(expr, var_id)|
                ExprTree::StackVarRewrite{
                    expr: expr.into(),
                    stack: stack.name.clone(),
                    var_id
                }
            ).chain(expr.post_process.into_iter()).collect(),
            post_process: vec![],
            expr: vec![],
            phantom: PhantomData
        }
    }
}
impl<T: FieldSized> Mut<T> {
    pub fn var<T2: FieldSized>(mut self, stack: &Stack, f: impl FnOnce(MutVariable<T>) -> Expr<T2>) -> Expr<T2> {
        let expr_count = self.expr.expr.len();
        let var_id: Vec<VarID> = (0..expr_count).map(|_| VarID::new()).collect();
        let mut ret = f(MutVariable{stack: stack.name.clone(), phantom: PhantomData, var_id: var_id.clone()});
        let mut statements = self.expr.statements;
        for (expr, var_id) in self.expr.expr.into_iter().zip(var_id.clone()) {
            statements.push(ExprTree::StackPush{
                var_id,
                stack: stack.name.clone(),
                expr: expr.into()
            });
        }
        statements.append(&mut self.expr.post_process);
        statements.append(&mut ret.statements);
        for var_id in var_id.into_iter().rev() {
            ret.post_process.push(ExprTree::StackDelete{stack: stack.name.clone(), var_id});
        }
        Expr {
            statements,
            post_process: ret.post_process,
            expr: ret.expr,
            phantom: PhantomData
        }
    }
}

impl Expr<()> {
    pub fn while_(mut cond: Expr<bool>, mut body: Expr<()>) -> Expr<()>{
        let mut body_trees = body.statements;
        body_trees.append(&mut body.post_process);
        Expr {
            statements: vec![ExprTree::While{
                cond_statements: cond.statements,
                cond_expr: cond.expr.pop().unwrap().into(),
                cond_post_process: cond.post_process,
                body: body_trees
            }],
            post_process: vec![],
            expr: vec![],
            phantom: PhantomData
        }
    }
}

pub struct HeapMemory<T: FieldSized> {
    main: Rc<str>,
    unused_memory: Rc<str>,
    phantom: PhantomData<T>
}
impl<T: FieldSized> HeapMemory<T> {
    pub fn new(main: Rc<str>, unused_memory: Rc<str>) -> Self {
        Self {
            main,
            unused_memory,
            phantom: PhantomData
        }
    }
    pub fn alloc(&self, expr: Expr<T>) -> Allocater<T> {
        Allocater{
            unused_memory: self.unused_memory.clone(),
            main: self.main.clone(),
            expr
        }
    }
}

pub struct Allocater<T: FieldSized> {
    main: Rc<str>,
    unused_memory: Rc<str>,
    expr: Expr<T>
}
impl<T: FieldSized> Allocater<T> {
    fn var<T2: FieldSized>(self, stack: &Stack, f: impl FnOnce(Expr<Mem<T>>) -> Expr<T2>) -> Expr<T2> {
        todo!()
    }
}

pub struct Mem<T: FieldSized> {
    phantom: PhantomData<T>
}
impl<T: FieldSized> FieldSized for Mem<T> {
    fn size() -> usize {
        1
    }
}

macro_rules! action {
    {$stack:expr; let! $v:pat = $e:expr; $(let! $v2:pat = $e2:expr;)* $e3:expr} => {
        $e.var(&$stack, |$v| action!{$stack; $(let! $v2 = $e2;)* $e3})
    };
    {$stack:expr; $e:expr} => ($e);
}

fn main(){
    let stack = Stack::new("stack");
    let expr = action!{
        stack;
        let! x = Expr::from(0.0).mut_();
        Expr::while_(x.get().equals(Expr::from(0.0)), x.rewrite(&stack, x.get() + Expr::from(1.0)))
    };
    println!("{:?}", expr);
    println!("{}", expr.create_tosh());
}
