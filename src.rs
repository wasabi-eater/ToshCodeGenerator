use core::ops::*;
use core::cell::{Cell, RefCell};
use std::marker::PhantomData;

struct StackDelete{
    stack: String
}
pub struct Expr<'a, T>{
    statements: Vec<String>,
    post_process: Vec<StackDelete>,
    expr: String,
    phantom: PhantomData<&'a T>
}
impl<'a, T> Expr<'a, T>{
    fn create_tosh(self) -> String {
        format!("{}\n{}", self.statements.join("\n"),
            self.post_process.into_iter().map(|s| format!("delete at 1 of {}", s.stack)).collect::<Vec<_>>().join("\n"))
    }
}
impl From<f64> for Expr<'static, f64> {
    fn from(n: f64) -> Self {
        Expr{
            statements: vec![],
            post_process: vec![],
            expr: n.to_string(),
            phantom: PhantomData
        }
    }
}
impl<'a> Neg for Expr<'a, f64>{
    type Output = Self;
    fn neg(self) -> Self {
        Expr{
            statements: self.statements,
            post_process: self.post_process,
            expr: format!("(-{})", self.expr),
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
            expr: format!("({} + {})", self.expr, other.expr),
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
            expr: format!("({} - {})", self.expr, other.expr),
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
            expr: format!("({} * {})", self.expr, other.expr),
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
            expr: format!("({} / {})", self.expr, other.expr),
            phantom: PhantomData
        }
    }
}
impl<'a> From<&'a str> for Expr<'static, String> {
    fn from(s: &str) -> Self {
        Expr {
            statements: vec![],
            post_process: vec![],
            expr: "\"".to_string() + s + "\"",
            phantom: PhantomData
        }
    }
}

pub struct Stack{
    stack: String,
    var_count: Cell<usize>
}
impl Stack{
    pub fn new(stack: impl Into<String>) -> Self {
        Self {stack: stack.into(), var_count: Cell::new(0)}
    }
    pub fn var<'a, 'b, T: 'b>(&'b self, expr: Expr<'a, T>, f: impl FnOnce(Variable<'b, T>) -> Expr<'a, T>) -> Expr<'a, T> {
        self.var_count.set(self.var_count.get() + 1);
        let number = self.var_count.get();
        let mut r = f(Variable{stack: &self, phantom: PhantomData, number});
        self.var_count.set(self.var_count.get() - 1);
        let mut statements = expr.statements;
        statements.push(format!("insert {} at 1 of {}", expr.expr, self.stack));
        for StackDelete{stack} in expr.post_process {
            if stack == self.stack {
                statements.push(format!("delete at 2 of {}", stack));
            }
            else {
                statements.push(format!("delete at 1 of {}", stack));
            }
        }
        statements.append(&mut r.statements);
        r.post_process.push(StackDelete{stack: self.stack.clone()});
        Expr {
            statements,
            post_process: r.post_process,
            expr: r.expr,
            phantom: PhantomData
        }
    }
}

pub struct Variable<'a, T>{
    stack: &'a Stack,
    phantom: PhantomData<&'a T>,
    number: usize
}
impl<'a, T> Variable<'a, T> {
    pub fn get(&self) -> Expr<'a, T>{
        Expr{
            statements: vec![],
            post_process: vec![],
            expr: format!("(item {} of {})", self.stack.var_count.get() - self.number + 1, self.stack.stack),
            phantom: PhantomData
        }
    }
}

macro_rules! action {
    {$stack:expr; let! $v:pat = $e:expr; $(let! $v2:pat = $e2:expr;)* $e3:expr} => {
        $stack.var($e, |$v| action!{$stack; $(let! $v2 = $e2;)* $e3})
    };
    {$stack:expr; $e:expr} => ($e);
}

fn main(){
    let stack = Stack::new("stack");
    let expr = action!{
        stack;
        let! z = action!{
            stack;
            let! x = Expr::from(0.4) + Expr::from(3.2);
            let! y = x.get() + Expr::from(3.0);
            x.get() * y.get()
        };
        z.get()
    };
    println!("{}", expr.create_tosh());
}
