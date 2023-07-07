use std::fmt;
use std::collections::HashMap;
use std::rc::Rc;
use core::ops::*;
use core::cell::Cell;
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
#[derive(Debug)]
enum ExprTree{
    Num(String),
    Str(String),
    BinOp{
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
    }
}
impl ExprTree{
    pub fn as_tosh_code(statements: impl Iterator<Item = ExprTree>) -> String{
        Self::create_tosh(statements, &mut HashMap::new())
    }
    fn create_tosh(statements: impl Iterator<Item = ExprTree>, stacks: &mut HashMap<Rc<str>, Vec<VarID>>) -> String{
        let mut code: Vec<String> = vec![];
        for expr in statements{
            let str = match expr {
                ExprTree::Num(n) => n,
                ExprTree::Str(s) => format!("\"{}\"", s),
                ExprTree::BinOp{left, op, right} => format!("({} {} {})",
                    Self::create_tosh(vec![*left].into_iter(), stacks),
                    op,
                    Self::create_tosh(vec![*right].into_iter(), stacks)),
                ExprTree::StackVar{var_id, stack: stack_name} => {
                    let stack = stacks.get(&stack_name).unwrap();
                    format!("(item {} of {})",
                        stack.len() - stack.iter().position(|id| *id == var_id).unwrap(),
                        stack_name
                    )
                },
                ExprTree::StackPush{var_id, stack: stack_name, expr} => {
                    let insertion = format!("insert {} at 1 of {}",
                        Self::create_tosh(vec![*expr].into_iter(), stacks),
                        stack_name
                    );
                    let stack = match stacks.get_mut(&stack_name) {
                        Some(stack) => stack,
                        None => {
                            stacks.insert(stack_name.clone(), vec![]);
                            stacks.get_mut(&stack_name).unwrap()
                        }
                    };
                    stack.push(var_id);
                    insertion
                },
                ExprTree::StackDelete{var_id, stack: stack_name} => {
                    let stack = stacks.get_mut(&stack_name).unwrap();
                    let pos = stack.iter().position(|id| *id == var_id).unwrap();
                    stack.remove(pos);
                    format!("delete {} of {}",
                        stack.len() - pos + 1,
                        stack_name
                    )
                }
            };
            code.push(str)
        }
        code.join("\n")
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
impl<T1: FieldSized, T2: FieldSized> FieldSized for (T1, T2) {
    fn size() -> usize {
        T1::size() + T2::size()
    }
}
#[derive(Debug)]
struct StackDelete{
    stack: Rc<str>,
    var_id: VarID
}
#[derive(Debug)]
pub struct Expr<'a, T : FieldSized>{
    statements: Vec<ExprTree>,
    post_process: Vec<StackDelete>,
    expr: Vec<ExprTree>,
    phantom: PhantomData<&'a T>
}
impl<'a, T : FieldSized> Expr<'a, T>{
    fn create_tosh(self) -> String {
        ExprTree::as_tosh_code(self.statements.into_iter().chain(
            self.post_process.into_iter().map(|s| ExprTree::StackDelete{
                var_id: s.var_id,
                stack: s.stack
            })))
    }
    pub fn var<'b, 'c, T2 : FieldSized>(self, stack: &'c Stack, f: impl FnOnce(Variable<'c, T>) -> Expr<'b, T2>) -> Expr<'b, T2>
        where T: 'c {
        let expr_count = self.expr.len();
        let var_id: Vec<VarID> = (0..expr_count).map(|_| VarID::new()).collect();
        let mut r = f(Variable{stack: stack.name.clone(), phantom: PhantomData, var_id: var_id.clone()});
        
        let mut statements = self.statements;
        for (expr, var_id) in self.expr.into_iter().zip(var_id.clone()) {
            statements.push(ExprTree::StackPush{
                var_id,
                stack: stack.name.clone(),
                expr: expr.into()
            });
        }
        for StackDelete{stack, var_id} in self.post_process {
            statements.push(ExprTree::StackDelete{
                var_id,
                stack
            });
        }
        statements.append(&mut r.statements);
        for var_id in var_id {
            r.post_process.push(StackDelete{stack: stack.name.clone(), var_id});
        }
        Expr {
            statements,
            post_process: r.post_process,
            expr: r.expr,
            phantom: PhantomData
        }
    }
}
impl<'a> From<()> for Expr<'a, ()> {
    fn from(_: ()) -> Self {
        Expr {
            statements: vec![],
            post_process: vec![],
            expr: vec![],
            phantom: PhantomData
        }
    }
}
impl <'a, T1: Into<Expr<'a, A>>, T2: Into<Expr<'a, B>>, A: FieldSized, B: FieldSized> From<(T1, T2)> for Expr<'a, (A, B)> {
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
impl <'a, T1: FieldSized, T2: FieldSized> Expr<'a, (T1, T2)> {
    pub fn item0(mut self) -> Expr<'a, T1> {
        let _ = self.expr.split_off(T1::size());
        Expr{
            statements: self.statements,
            expr: self.expr,
            post_process: self.post_process,
            phantom: PhantomData
        }
    }
    pub fn item1(mut self) -> Expr<'a, T2> {
        let t2_expr = self.expr.split_off(T1::size());
        Expr{
            statements: self.statements,
            expr: t2_expr,
            post_process: self.post_process,
            phantom: PhantomData
        }
    }
}
impl<'a> From<f64> for Expr<'a, f64> {
    fn from(n: f64) -> Self {
        Expr{
            statements: vec![],
            post_process: vec![],
            expr: vec![ExprTree::Num(n.to_string())],
            phantom: PhantomData
        }
    }
}
impl<'a> Neg for Expr<'a, f64>{
    type Output = Self;
    fn neg(mut self) -> Self {
        Expr{
            statements: self.statements,
            post_process: self.post_process,
            expr: vec![ExprTree::BinOp{
                left: ExprTree::Num("0".into()).into(),
                op: "-",
                right: self.expr.pop().unwrap().into()
            }],
            phantom: PhantomData
        }
    }
}
impl<'a> Add for Expr<'a, f64> {
    type Output = Self;
    fn add(mut self, mut other: Self) -> Self {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        let mut post_process = other.post_process;
        post_process.append(&mut self.post_process);
        Expr {
            statements,
            post_process,
            expr: vec![ExprTree::BinOp{
                left: self.expr.pop().unwrap().into(),
                op: "+",
                right: other.expr.pop().unwrap().into()
            }],
            phantom: PhantomData
        }
    }
}
impl<'a> Sub for Expr<'a, f64> {
    type Output = Self;
    fn sub(mut self, mut other: Self) -> Self {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        let mut post_process = other.post_process;
        post_process.append(&mut self.post_process);
        Expr {
            statements,
            post_process,
            expr: vec![ExprTree::BinOp{
                left: self.expr.pop().unwrap().into(),
                op: "-",
                right: other.expr.pop().unwrap().into()
            }],
            phantom: PhantomData
        }
    }
}
impl<'a> Mul for Expr<'a, f64> {
    type Output = Self;
    fn mul(mut self, mut other: Self) -> Self {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        let mut post_process = other.post_process;
        post_process.append(&mut self.post_process);
        Expr {
            statements,
            post_process,
            expr: vec![ExprTree::BinOp{
                left: self.expr.pop().unwrap().into(),
                op: "*",
                right: other.expr.pop().unwrap().into()
            }],
            phantom: PhantomData
        }
    }
}
impl<'a> Div for Expr<'a, f64> {
    type Output = Self;
    fn div(mut self, mut other: Self) -> Self {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        let mut post_process = other.post_process;
        post_process.append(&mut self.post_process);
        Expr {
            statements,
            post_process,
            expr: vec![ExprTree::BinOp{
                left: self.expr.pop().unwrap().into(),
                op: "/",
                right: other.expr.pop().unwrap().into()
            }],
            phantom: PhantomData
        }
    }
}
impl<'a, 'b> From<&'a str> for Expr<'b, String> {
    fn from(s: &str) -> Self {
        Expr {
            statements: vec![],
            post_process: vec![],
            expr: vec![ExprTree::Str(s.into())],
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

pub struct Variable<'a, T : FieldSized>{
    stack: Rc<str>,
    phantom: PhantomData<&'a T>,
    var_id: Vec<VarID>
}
impl<'a, T : FieldSized> Variable<'a, T> {
    pub fn get(&self) -> Expr<'a, T>{
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
        let! x = Expr::from(0.0);
        let! y = action!{
            stack;
            let! z = x.get() + Expr::from(3.2);
            let! w = z.get() * Expr::from(2.0);
            Expr::from((w.get(), "Hey"))
        };
        let! _ = Expr::from((y.get(), "Yeah!"));
        Expr::from(())
    };
    println!("{:?}", expr);
    println!("{}", expr.create_tosh());
}
