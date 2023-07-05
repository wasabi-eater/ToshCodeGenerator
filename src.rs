use core::ops::*;
use core::cell::Cell;
use std::marker::PhantomData;

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
struct StackDelete{
    stack: String
}
pub struct Expr<'a, T : FieldSized>{
    statements: Vec<String>,
    post_process: Vec<StackDelete>,
    expr: Vec<String>,
    phantom: PhantomData<&'a T>
}
impl<'a, T : FieldSized> Expr<'a, T>{
    fn create_tosh(self) -> String {
        format!("{}\n{}", self.statements.join("\n"),
            self.post_process.into_iter().map(|s| format!("delete at 1 of {}", s.stack)).collect::<Vec<_>>().join("\n"))
    }
    pub fn var<'b, 'c, T2 : FieldSized>(self, stack: &'c Stack, f: impl FnOnce(Variable<'c, T>) -> Expr<'b, T2>) -> Expr<'b, T2>
        where T: 'c {
        let expr_count = self.expr.len();
        let number = (stack.var_count.get() + 1 .. stack.var_count.get() + expr_count + 1).collect();
        stack.var_count.set(stack.var_count.get() + expr_count);
        let mut r = f(Variable{stack: &stack, phantom: PhantomData, number});
        stack.var_count.set(stack.var_count.get() - 1);
        let mut statements = self.statements;
        for expr in self.expr {
            statements.push(format!("insert {} at 1 of {}", expr, stack.stack));
        }
        for StackDelete{stack: _stack} in self.post_process {
            if _stack == stack.stack {
                statements.push(format!("delete at {} of {}", expr_count + 1, _stack));
            }
            else {
                statements.push(format!("delete at 1 of {}", _stack));
            }
        }
        statements.append(&mut r.statements);
        r.post_process.push(StackDelete{stack: stack.stack.clone()});
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
impl<'a> From<f64> for Expr<'a, f64> {
    fn from(n: f64) -> Self {
        Expr{
            statements: vec![],
            post_process: vec![],
            expr: vec![n.to_string()],
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
            expr: vec![format!("(-{})", self.expr[0])],
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
            expr: vec![format!("({} + {})", self.expr[0], other.expr[0])],
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
            expr: vec![format!("({} - {})", self.expr[0], other.expr[0])],
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
            expr: vec![format!("({} * {})", self.expr[0], other.expr[0])],
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
            expr: vec![format!("({} / {})", self.expr[0], other.expr[0])],
            phantom: PhantomData
        }
    }
}
impl<'a, 'b> From<&'a str> for Expr<'b, String> {
    fn from(s: &str) -> Self {
        Expr {
            statements: vec![],
            post_process: vec![],
            expr: vec!["\"".to_string() + s + "\""],
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
}

pub struct Variable<'a, T : FieldSized>{
    stack: &'a Stack,
    phantom: PhantomData<&'a T>,
    number: Vec<usize>
}
impl<'a, T : FieldSized> Variable<'a, T> {
    pub fn get(&self) -> Expr<'a, T>{
        Expr{
            statements: vec![],
            post_process: vec![],
            expr: self.number.iter()
                .map(|number| format!("(item {} of {})", self.stack.var_count.get() - number + 1, self.stack.stack))
                .collect(),
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
        let! z = action!{
            stack;
            let! x = Expr::from(0.4) + Expr::from(3.2);
            let! y = x.get() + Expr::from(3.0);
            x.get() * y.get()
        };
        let! w = Expr::from((z.get(), 3.2));
        w.get()
    };
    println!("{}", expr.create_tosh());
}
