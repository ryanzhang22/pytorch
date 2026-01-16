#include <c10/core/Scalar.h>
#include <ATen/core/TensorBody.h>
#include <string_view>

namespace at {

namespace {

// Verifies the requested type is the same as the Tensor's type.
void check_type(const TensorBase& tensor, ScalarType type, std::string_view type_name) {
  TORCH_CHECK(
      tensor.scalar_type() == type
      || (isQIntType(tensor.scalar_type())
          && toUnderlying(tensor.scalar_type()) == type),
      "expected scalar type ", type_name, " but found ", tensor.scalar_type());
}

} // namespace

#define DEFINE_CAST(T, name)                                         \
   template <>                                                       \
   TORCH_API const T* TensorBase::const_data_ptr() const {           \
     check_type(*this, ScalarType::name, #name);                     \
     return this->unsafeGetTensorImpl()->data_ptr_impl<T>();         \
   }                                                                 \
                                                                     \
   template <>                                                       \
   TORCH_API const T* TensorBase::const_data_ptr<const T>() const {  \
     check_type(*this, ScalarType::name, #name);                     \
     return this->unsafeGetTensorImpl()->data_ptr_impl<std::remove_const_t<T>>(); \
   }                                                                 \
                                                                     \
   template <>                                                       \
   TORCH_API T* TensorBase::mutable_data_ptr() const {               \
     check_type(*this, ScalarType::name, #name);                     \
     return this->unsafeGetTensorImpl()->mutable_data_ptr_impl<T>(); \
   }                                                                 \
                                                                     \
   template <>                                                       \
   TORCH_API T* TensorBase::data_ptr() const {                       \
     return mutable_data_ptr<T>();                                   \
   }                                                                 \

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CAST)
 AT_FORALL_QINT_TYPES(DEFINE_CAST)
 DEFINE_CAST(uint16_t, UInt16)
 DEFINE_CAST(uint32_t, UInt32)
 DEFINE_CAST(uint64_t, UInt64)
 #undef DEFINE_CAST

 #define DEFINE_ITEM(T, name)      \
   template <>                     \
   TORCH_API T Tensor::item() const { \
     return item().to##name();     \
   }

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ITEM)
 #undef DEFINE_ITEM

 } //namespace at

#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>

namespace at {
static Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  if (strides.empty()) {
      return at::detail::empty_cpu(sizes, options);
  } else {
      return at::detail::empty_strided_cpu(sizes, strides, options);
  }
}

struct structured_mul_out_functional final : public at::native::structured_mul_out {
    void set_output_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_mul_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    void set_output_raw_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_mul_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return outputs_[output_idx];
    }
    std::array<Tensor, 1> outputs_;
};

static void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {
  // These checks are needed on those operators that:
  //   1) don't use 'TensorIterator' (e.g. 'addmm' and 'baddbmm')
  //   2) have particular typing rules (e.g. 'cumsum' and 'cumprod')
  // For other operators (e.g. 'add'), 'TensorIterator' already checks
  // these things separately.
  TORCH_CHECK(options.dtype() == self.dtype(),
      "Bad in-place call: ",
      "input tensor dtype ", self.dtype(), " and output tensor dtype ", options.dtype(), " should match");
  TORCH_CHECK(options.device() == self.device(),
      "Bad in-place call: ",
      "input tensor device ", self.device(), " and output tensor device ", options.device(), " should match");
  TORCH_CHECK(sizes == self.sizes(),
      "Bad in-place call: ",
      "input tensor size ", self.sizes(), " and output tensor size ", sizes, " should match");
}

static std::optional<Tensor> maybe_create_proxy(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  if (out.strides() != strides) {
    return at::detail::empty_strided_cpu(sizes, strides, options);
  }
  return std::nullopt;
}

struct structured_mul_out_inplace final : public at::native::structured_mul_out {
    structured_mul_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
    void set_output_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
        if (C10_UNLIKELY(maybe_proxy.has_value())) {
            proxy_outputs_[output_idx] = std::move(maybe_proxy).value();
        }
        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_mul_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    void set_output_raw_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        const auto& out = outputs_[output_idx].get();
        check_inplace(out, sizes, options);
        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_mul_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get();
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
    std::array<::std::optional<Tensor>, 1> proxy_outputs_;
};

Tensor Tensor::mul(const Tensor& other) const {
  structured_mul_out_functional op;
  op.meta(*this, other);
  op.impl(*this, other, op.maybe_get_output(0));
  return std::move(op.outputs_[0]);
}

Tensor& Tensor::mul_(const Tensor& other) const {
  structured_mul_out_inplace op(const_cast<Tensor&>(*this));
  op.meta(*this, other);
  op.impl(*this, other, op.maybe_get_output(0));
  return const_cast<Tensor&>(*this);
}

}
